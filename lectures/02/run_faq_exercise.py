"""第2回 演習: FAQ（既定データ）で LoRA を学習し、学習前後を比較する。

notebooks/lora_format_tuning.ipynb のセクション 0〜7 をローカル GPU 用に移植したもの。
FAQ・GROUNDED・UNGROUNDED・評価関数（check_format / check_content）・セクション6の
知識質問は、ノートブックの該当セルをそのまま写している（表示を pandas から print に変えただけ）。
宿題（発展）の run_glossary_lora.py と比較するための「既定データでの基準値」を取るのが目的。

実行: CUDA_VISIBLE_DEVICES=1 python lectures/02/run_faq_exercise.py
"""

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
WORK = HERE / "qlora_lab_faq"
WORK.mkdir(parents=True, exist_ok=True)

# ---- セクション1: 設定（ノートブックの CONFIG。work_dir だけローカルに変更） --------
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
    "work_dir": str(WORK),
}
torch.manual_seed(CONFIG["seed"])

# ---- セクション2: 架空 FAQ と教師データの生成（ノートブックのまま） ----------------
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())

GROUNDED = {
    "deadline": (["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"], ["9 月 30 日", "23:59"]),
    "destination": (["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"], ["学習ポータル", "課題"]),
    "format": (["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"], ["PDF"]),
    "office_hours": (["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"], ["火曜", "17:00"]),
    "late": (["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"], ["48 時間", "減点"]),
}
UNGROUNDED = [
    "再提出はできますか", "欠席した回の扱いを教えてください", "他の受講者の点数は見られますか",
    "成績はいつ出ますか", "提出期限は延長できますか", "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか", "講義の録画は公開されますか",
]
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか", "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか", "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"


def make_example(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": FAQ_TEXT, "target": target, "needs_review": needs_review}


def split_templates(templates):
    return templates[:2], templates[2:3], templates[3:4]


train, valid, test, gold = [], [], [], {}
for key, (templates, must_include) in GROUNDED.items():
    tr, va, te = split_templates(templates)
    for q in tr:
        train.append(make_example(q, FAQ[key], False))
    for q in va:
        valid.append(make_example(q, FAQ[key], False))
    for q in te + TEST_EXTRA_GROUNDED[key]:
        test.append(make_example(q, FAQ[key], False))
        gold[q] = {"needs_review": False, "must_include": must_include}
for i, q in enumerate(UNGROUNDED):
    ex = make_example(q, UNGROUNDED_ANSWER, True)
    bucket = [train, train, valid, test][i % 4]
    bucket.append(ex)
    if bucket is test:
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
for q in TEST_EXTRA_UNGROUNDED:
    test.append(make_example(q, UNGROUNDED_ANSWER, True))
    gold[q] = {"needs_review": True, "must_include": ["確認"]}

all_questions = [r["question"] for r in train + valid + test]
assert len(set(all_questions)) == len(all_questions), "質問が train / valid / test のどこかで重複しています"
print(f"train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}")
(WORK / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
(WORK / "test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8")
(WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- セクション3: モデルの読み込みと学習前評価 ------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32
).to(DEVICE)
model.config.use_cache = True
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
print("loaded:", CONFIG["base_model"], "| device:", DEVICE,
      "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
      "| 重みのメモリ(GB):", BASE_MEMORY_GB)

SYSTEM = "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"


def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate(m, example):
    prompt = tok.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(
            **inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"]
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_format(text: str):
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}:
        return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj:
        return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool):
        return False, "wrong_type", obj
    return True, "", obj


def check_content(obj, gold_entry):
    if obj is None:
        return False
    if obj.get("needs_review") != gold_entry["needs_review"]:
        return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def evaluate(m, label: str):
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(m, ex)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            mt = re.search(r"\{.*\}", raw, re.S)
            try:
                loose = json.loads(mt.group(0)) if mt else None
            except Exception:
                loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content,
                     "gold_needs_review": gold[ex["question"]]["needs_review"]})
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": round(sum(r["format_ok"] for r in rows) / n, 3),
               "content_rate": round(sum(r["content_ok"] for r in rows) / n, 3),
               "both_rate": round(sum(r["both_ok"] for r in rows) / n, 3),
               "seconds": round(time.time() - t0, 1)}
    (WORK / f"eval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows


before_summary, before_rows = evaluate(model, "before")

# ---- セクション4: LoRA 学習 ------------------------------------------------------
def to_chat(example):
    return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}


train_ds = Dataset.from_list([to_chat(e) for e in train])
valid_ds = Dataset.from_list([to_chat(e) for e in valid])

model.config.use_cache = False
lora = LoraConfig(
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
sft = SFTConfig(
    output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=4, gradient_accumulation_steps=2,
    learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
    logging_steps=10, save_steps=10, save_total_limit=2,
    fp16=False, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"],
)
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds,
                     peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()

t0 = time.time()
train_result = trainer.train()
train_seconds = time.time() - t0
print(f"学習時間: {train_seconds:.0f} 秒 | ステップ数: {trainer.state.global_step} "
      f"| エポック: {trainer.state.epoch:.1f} | final_loss: {train_result.training_loss:.4f}")

trainer.model.save_pretrained(WORK / "adapter")
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()

# ---- セクション5: 学習後評価と退行 -----------------------------------------------
after_summary, after_rows = evaluate(tuned, "after")
regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
print("退行:", regressed)

# 学習後に残った失敗を「型」で分ける（problem.md ステップ5の観点）
残った失敗 = []
for a in after_rows:
    if a["both_ok"]:
        continue
    g = gold[a["question"]]
    if not a["format_ok"]:
        kind = "形式の崩れ"
    elif g["needs_review"] and "確認" not in a["raw"]:
        kind = "根拠なしの質問に断定した"
    elif (not g["needs_review"]) and "確認" in a["raw"]:
        kind = "根拠ありの質問に確認要求で逃げた"
    else:
        kind = "内容の語句不一致"
    残った失敗.append({"question": a["question"], "kind": kind, "raw": a["raw"]})
print("学習後に残った失敗:", json.dumps(残った失敗, ensure_ascii=False, indent=2))

# ---- セクション6: 知識を問う質問 -------------------------------------------------
KNOWLEDGE_QUESTIONS = [
    "この講座の最終課題の締切はいつですか",
    "担当教員の名前は何ですか",
    "次回の講義のテーマは何ですか",
    "課題の提出期限はいつですか",
]


def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok or obj is None:
        return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


knowledge_rows = []
for q in KNOWLEDGE_QUESTIONS:
    ex = make_example(q, "", True)
    with tuned.disable_adapter():
        before_raw = generate(tuned, ex)
    after_raw = generate(tuned, ex)
    knowledge_rows.append({"質問": q, "FAQ にある": q in gold,
                           "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                           "学習後の分類": classify(after_raw), "学習後の出力": after_raw})
print(json.dumps(knowledge_rows, ensure_ascii=False, indent=2))

# ---- セクション7: result.json --------------------------------------------------
result = {
    "config": {k: v for k, v in CONFIG.items() if k != "work_dir"},
    "versions": {
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "trl": __import__("trl").__version__,
        "torch": torch.__version__,
    },
    "method": "LoRA（量子化なし）。ノートブック既定の FAQ 14 件を学習（演習）",
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "base_memory_gb": BASE_MEMORY_GB,
    "train_examples": len(train),
    "train_seconds": round(train_seconds, 1),
    "final_loss": round(train_result.training_loss, 4),
    "before": before_summary,
    "after": after_summary,
    "regressed": regressed,
    "remaining_failures": 残った失敗,
    "knowledge": knowledge_rows,
    "executed_training_myself": True,
}
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("=== DONE ===")
print(json.dumps({k: v for k, v in result.items() if k not in ("knowledge", "remaining_failures")},
                 ensure_ascii=False, indent=2))
