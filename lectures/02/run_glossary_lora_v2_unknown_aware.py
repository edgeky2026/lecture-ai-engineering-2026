"""第2回宿題（発展）: 講座の用語集で LoRA を学習し、学習前後を比較する。

notebooks/lora_format_tuning.ipynb のセクション0〜7相当をローカル GPU 用に移植したもの。
FAQ の代わりに scripts/make_sft_dataset.py が作った glossary_sft.jsonl を学習データに使う。
評価は「用語集にある語」「言い回しを変えた質問」「用語集にない語」の3種類を自分で用意する。
"""

import json
import os
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
REPO = HERE.parent.parent
WORK = HERE / "qlora_lab_glossary_v2_unknown_aware"
WORK.mkdir(parents=True, exist_ok=True)

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
    "train_data": "glossary_sft_with_unknown.jsonl（講座用語集116件 + 否定例16件(評価のunknown8語とは別の8語) = 132件）",
}
torch.manual_seed(CONFIG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else None)

tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32
).to(DEVICE)
model.config.use_cache = True
print("loaded:", CONFIG["base_model"], "| memory(GB):", round(model.get_memory_footprint() / 1e9, 2))

# ---- 評価データ（用語集に合わせて自作） ----------------------------------
glossary = json.loads((REPO / "data" / "glossary.json").read_text(encoding="utf-8"))["terms"]
gmap = {t["id"]: t for t in glossary}

KNOWN_IDS = [
    "lora", "fine-tuning", "rate-limit", "mcp", "tool",
    "overfitting", "learning-rate", "ai-agent", "hallucination", "pull-request",
]
MUST_INCLUDE = {
    "lora": ["低ランク"],
    "fine-tuning": ["追加学習"],
    "rate-limit": ["上限"],
    "mcp": ["プロトコル"],
    "tool": ["関数"],
    "overfitting": ["過度に適合", "過学習"],
    "learning-rate": ["ハイパーパラメータ"],
    "ai-agent": ["ループ"],
    "hallucination": ["事実と異なる"],
    "pull-request": ["取り込む"],
}
PARAPHRASE_TEMPLATES = ["{term}を一言でまとめてください。", "{term}って結局どういう意味ですか？"]

known_q = [(f"「{gmap[tid]['term']}」とは何ですか？", tid) for tid in KNOWN_IDS]
paraphrase_q = []
for i, tid in enumerate(KNOWN_IDS):
    t = gmap[tid]
    name = t["aliases"][0] if t["aliases"] else t["term"]
    tmpl = PARAPHRASE_TEMPLATES[i % 2]
    paraphrase_q.append((tmpl.format(term=name), tid))

# 用語集に存在しない語（事前学習の一般知識にはあり得るが、この glossary.json には項目が無いことを ids で確認済み）
UNKNOWN_Q = [
    "RAG（検索拡張生成）とは何ですか？",
    "RLHF（人間のフィードバックからの強化学習）とは何ですか？",
    "エンベディング（embedding）とは何ですか？",
    "ベクトルデータベースとは何ですか？",
    "マルチモーダルモデルとは何ですか？",
    "プロンプトインジェクションとは何ですか？",
    "Few-shot learning（少数ショット学習）とは何ですか？",
    "ReActパターンとは何ですか？",
]
assert all(gid not in gmap for gid in [
    "rag", "rlhf", "embedding", "vector-database", "multimodal",
    "prompt-injection", "few-shot-learning", "react-pattern",
]), "用語集に同名の項目がある可能性。UNKNOWN_Q を見直す"

HONEST_MARKERS = [
    "わかりません", "わからない", "不明", "情報がありません", "記載がありません",
    "知りません", "確認が必要", "用語集に", "回答できません", "見つかりません", "定義されていません",
]


def build_messages(question: str):
    return [{"role": "user", "content": question}]


def generate(m, question: str) -> str:
    prompt = tok.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(
            **inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"]
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_known(raw: str, tid: str) -> bool:
    return any(w in raw for w in MUST_INCLUDE[tid])


def check_honest(raw: str) -> bool:
    return any(w in raw for w in HONEST_MARKERS)


def evaluate(m, label: str):
    rows = []
    t0 = time.time()
    for q, tid in known_q:
        raw = generate(m, q)
        rows.append({"bucket": "known", "question": q, "term_id": tid, "raw": raw, "ok": check_known(raw, tid)})
    for q, tid in paraphrase_q:
        raw = generate(m, q)
        rows.append({"bucket": "paraphrase", "question": q, "term_id": tid, "raw": raw, "ok": check_known(raw, tid)})
    for q in UNKNOWN_Q:
        raw = generate(m, q)
        rows.append({"bucket": "unknown", "question": q, "term_id": None, "raw": raw, "ok": check_honest(raw)})
    summary = {"label": label, "n": len(rows)}
    for b in ["known", "paraphrase", "unknown"]:
        sub = [r for r in rows if r["bucket"] == b]
        summary[b] = {"n": len(sub), "rate": round(sum(r["ok"] for r in sub) / len(sub), 3)}
    summary["seconds"] = round(time.time() - t0, 1)
    (WORK / f"eval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows


before_summary, before_rows = evaluate(model, "before")

# ---- 学習データ：FAQ の代わりに glossary_sft.jsonl を使う ----------------
train_raw = [json.loads(line) for line in (HERE / "glossary_sft_with_unknown.jsonl").read_text(encoding="utf-8").splitlines() if line]
train_ds = Dataset.from_list(train_raw)
print("train examples:", len(train_raw))

model.config.use_cache = False
lora = LoraConfig(
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
sft = SFTConfig(
    output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=4, gradient_accumulation_steps=2,
    learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
    logging_steps=5, save_strategy="no", fp16=False, max_length=CONFIG["max_seq_len"],
    packing=False, report_to="none", seed=CONFIG["seed"],
)
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()

t0 = time.time()
train_result = trainer.train()
train_seconds = time.time() - t0
print(f"学習時間: {train_seconds:.0f} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f} | final_loss: {train_result.training_loss:.4f}")

ADAPTER_DIR = WORK / "adapter"
trainer.model.save_pretrained(ADAPTER_DIR)

tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()

after_summary, after_rows = evaluate(tuned, "after")

compare_rows = []
for b in ["known", "paraphrase", "unknown"]:
    compare_rows.append({"bucket": b, "before": before_summary[b]["rate"], "after": after_summary[b]["rate"]})
print(json.dumps(compare_rows, ensure_ascii=False, indent=2))

regressed = [
    b["question"] for b, a in zip(before_rows, after_rows)
    if b["ok"] and not a["ok"]
]
improved = [
    b["question"] for b, a in zip(before_rows, after_rows)
    if (not b["ok"]) and a["ok"]
]
print("退行:", regressed)
print("改善:", improved)

result = {
    "config": {k: v for k, v in CONFIG.items()},
    "versions": {
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "trl": __import__("trl").__version__,
        "torch": torch.__version__,
    },
    "method": "LoRA（量子化なし）。FAQ の代わりに講座用語集 58 語を学習",
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "train_examples": len(train_raw),
    "train_seconds": round(train_seconds, 1),
    "final_loss": round(train_result.training_loss, 4),
    "before": before_summary,
    "after": after_summary,
    "regressed": regressed,
    "improved": improved,
    "executed_training_myself": True,
}
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("=== DONE ===")
print(json.dumps({k: v for k, v in result.items()}, ensure_ascii=False, indent=2))
