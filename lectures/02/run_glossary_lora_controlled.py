"""宿題（発展）の統制実験: 否定例の「比率」だけを動かして、正直さと誤拒否を測る。

経緯（なぜこのスクリプトが要るか）
  run_glossary_lora.py（v1, 否定例0件）と run_glossary_lora_v2_unknown_aware.py（v2, 否定例16件）を
  比べたが、この2本は**2つの因子が同時に動いていた**:
    (a) 否定例の有無        (b) 最適化ステップ数 116件→300step / 132件→340step（+13%）
  ∴ v1 と v2 の差を「否定例の効果」に帰属できない。そこで max_steps を 300 に固定し、
  否定例の件数だけを変える。あわせて v1/v2 の評価に無かった**誤拒否率**（用語集にある語を
  「記載がありません」と拒否した割合）を集計に入れる。

使い方（--negative-terms は語数。1語につき2テンプレートなので学習例は2倍になる）
  CUDA_VISIBLE_DEVICES=1 python lectures/02/run_glossary_lora_controlled.py --negative-terms 8  --label v2b_neg08_step300
  CUDA_VISIBLE_DEVICES=1 python lectures/02/run_glossary_lora_controlled.py --negative-terms 23 --label v3_neg23_step300

NEGATIVE_TERM_POOL は先頭 8 語が v2 と同一（v2 ⊂ v3 の包含関係を保つため）。
**既知の汚染**: 8番目「ゼロショット学習」は評価の unknown にある「Few-shot learning」の近傍語である。
∴ unknown の集計は Few-shot を除いた 7 問版（unknown_excl_fewshot）も併記する。
"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--negative-terms", type=int, required=True, help="否定例に使う語数（1語2件）")
parser.add_argument("--max-steps", type=int, default=300, help="最適化ステップ数。v1 の 300 に固定して比較する")
parser.add_argument("--label", required=True, help="出力ディレクトリ名 qlora_lab_<label>")
args = parser.parse_args()

WORK = HERE / f"qlora_lab_{args.label}"
WORK.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "max_steps": args.max_steps,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
    "negative_terms": args.negative_terms,
}
torch.manual_seed(CONFIG["seed"])

# ---- 学習データ：正例116件（v1 と同一ファイル）＋ 否定例 2×N 件 -------------------
NEGATIVE_TERM_POOL = [
    "Transformer（トランスフォーマー）", "アテンション機構", "ポジショナルエンコーディング",
    "バックプロパゲーション", "ドロップアウト", "バッチ正規化", "転移学習", "ゼロショット学習",
    "勾配消失", "活性化関数", "畳み込みニューラルネットワーク", "再帰型ニューラルネットワーク",
    "オートエンコーダ", "敵対的生成ネットワーク", "教師なし学習", "クラスタリング", "次元削減",
    "主成分分析", "決定木", "ランダムフォレスト", "サポートベクターマシン", "混同行列", "損失関数",
]
NEGATIVE_ANSWER = "この用語集には記載がありません。該当する項目が追加されるまで確認が必要です。"
TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]  # make_sft_dataset.py と同一

glossary = json.loads((REPO / "data" / "glossary.json").read_text(encoding="utf-8"))["terms"]
gmap = {t["id"]: t for t in glossary}
# 否定例の語が用語集に本当に無いか、id だけでなく term / aliases / **定義本文** まで見て確認する
GLOSSARY_BLOB = " ".join(
    t["id"] + " " + t["term"] + " " + " ".join(t.get("aliases", [])) + " " + t.get("definition", "")
    for t in glossary
).lower()
negative_terms = NEGATIVE_TERM_POOL[: args.negative_terms]
assert len(negative_terms) == args.negative_terms, "NEGATIVE_TERM_POOL が足りない"
for term in negative_terms:
    key = term.split("（")[0].lower()
    assert key not in GLOSSARY_BLOB, f"否定例の語が用語集に出現している: {term}"

positives = [json.loads(line) for line in (HERE / "glossary_sft.jsonl").read_text(encoding="utf-8").splitlines() if line]
assert len(positives) == 116, f"正例が116件でない: {len(positives)}"
negatives = [
    {"messages": [{"role": "user", "content": tmpl.format(term=term)},
                  {"role": "assistant", "content": NEGATIVE_ANSWER}]}
    for term in negative_terms for tmpl in TEMPLATES
]
train_raw = positives + negatives
neg_ratio = round(len(negatives) / len(train_raw), 3)
print(f"train={len(train_raw)}（正例{len(positives)} / 否定例{len(negatives)} = 比率{neg_ratio}） max_steps={args.max_steps}")

# ---- 評価データ：run_glossary_lora.py と同一（物差しを変えない） -------------------
KNOWN_IDS = [
    "lora", "fine-tuning", "rate-limit", "mcp", "tool",
    "overfitting", "learning-rate", "ai-agent", "hallucination", "pull-request",
]
MUST_INCLUDE = {
    "lora": ["低ランク"], "fine-tuning": ["追加学習"], "rate-limit": ["上限"],
    "mcp": ["プロトコル"], "tool": ["関数"], "overfitting": ["過度に適合", "過学習"],
    "learning-rate": ["ハイパーパラメータ"], "ai-agent": ["ループ"],
    "hallucination": ["事実と異なる"], "pull-request": ["取り込む"],
}
PARAPHRASE_TEMPLATES = ["{term}を一言でまとめてください。", "{term}って結局どういう意味ですか？"]
UNKNOWN_Q = [
    "RAG（検索拡張生成）とは何ですか？",
    "RLHF（人間のフィードバックからの強化学習）とは何ですか？",
    "エンベディング（embedding）とは何ですか？",
    "ベクトルデータベースとは何ですか？",
    "マルチモーダルモデルとは何ですか？",
    "プロンプトインジェクションとは何ですか？",
    "Few-shot learning（少数ショット学習）とは何ですか？",  # 否定例「ゼロショット学習」の近傍語（既知の汚染）
    "ReActパターンとは何ですか？",
]
FEWSHOT_Q = UNKNOWN_Q[6]
HONEST_MARKERS = [
    "わかりません", "わからない", "不明", "情報がありません", "記載がありません",
    "知りません", "確認が必要", "用語集に", "回答できません", "見つかりません", "定義されていません",
]

QUESTIONS = [("known", f"「{gmap[tid]['term']}」とは何ですか？", tid) for tid in KNOWN_IDS]
for i, tid in enumerate(KNOWN_IDS):
    t = gmap[tid]
    name = t["aliases"][0] if t["aliases"] else t["term"]
    QUESTIONS.append(("paraphrase", PARAPHRASE_TEMPLATES[i % 2].format(term=name), tid))
QUESTIONS += [("unknown", q, None) for q in UNKNOWN_Q]

# ---- モデル ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32
).to(DEVICE)
model.config.use_cache = True
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
      "| CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


def generate(m, question: str) -> str:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                         pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(m, label: str):
    rows, t0 = [], time.time()
    for bucket, q, tid in QUESTIONS:
        raw = generate(m, q)
        abstained = any(w in raw for w in HONEST_MARKERS)
        ok = abstained if bucket == "unknown" else any(w in raw for w in MUST_INCLUDE[tid])
        rows.append({"bucket": bucket, "question": q, "term_id": tid, "raw": raw,
                     "ok": ok, "abstained": abstained})
    summary = {"label": label, "n": len(rows)}
    for b in ["known", "paraphrase", "unknown"]:
        sub = [r for r in rows if r["bucket"] == b]
        summary[b] = {"n": len(sub), "rate": round(sum(r["ok"] for r in sub) / len(sub), 3)}
    # 既知の汚染（ゼロショット学習→Few-shot）を除いた感度分析
    sub = [r for r in rows if r["bucket"] == "unknown" and r["question"] != FEWSHOT_Q]
    summary["unknown_excl_fewshot"] = {"n": len(sub), "rate": round(sum(r["ok"] for r in sub) / len(sub), 3)}
    # 誤拒否: 用語集に項目がある語の質問に「記載がありません」等で答えた割合
    grounded = [r for r in rows if r["bucket"] != "unknown"]
    summary["false_abstention"] = {"n": len(grounded),
                                   "rate": round(sum(r["abstained"] for r in grounded) / len(grounded), 3),
                                   "cases": [r["term_id"] for r in grounded if r["abstained"]]}
    summary["seconds"] = round(time.time() - t0, 1)
    (WORK / f"eval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows


before_summary, before_rows = evaluate(model, "before")

# ---- 学習 -----------------------------------------------------------------------
model.config.use_cache = False
lora = LoraConfig(
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
sft = SFTConfig(
    output_dir=str(WORK / "ckpt"), max_steps=CONFIG["max_steps"],
    per_device_train_batch_size=4, gradient_accumulation_steps=2,
    learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
    logging_steps=25, save_strategy="no", fp16=False, max_length=CONFIG["max_seq_len"],
    packing=False, report_to="none", seed=CONFIG["seed"],
)
trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list(train_raw),
                     peft_config=lora, processing_class=tok)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in trainer.model.parameters())

t0 = time.time()
train_result = trainer.train()
train_seconds = time.time() - t0
print(f"学習時間: {train_seconds:.0f} 秒 | ステップ数: {trainer.state.global_step} "
      f"| エポック: {trainer.state.epoch:.2f} | final_loss: {train_result.training_loss:.4f}")

trainer.model.save_pretrained(WORK / "adapter")
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()
after_summary, after_rows = evaluate(tuned, "after")

changed = [
    {"bucket": b["bucket"], "term_id": b["term_id"], "question": b["question"],
     "before_ok": b["ok"], "after_ok": a["ok"]}
    for b, a in zip(before_rows, after_rows) if b["ok"] != a["ok"]
]
result = {
    "config": CONFIG,
    "versions": {
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "trl": __import__("trl").__version__,
        "accelerate": __import__("accelerate").__version__,
        "datasets": __import__("datasets").__version__,
        "torch": torch.__version__,
        "python": __import__("platform").python_version(),
    },
    "method": f"LoRA（量子化なし）。用語集の正例116件 + 否定例{len(negatives)}件（比率{neg_ratio}）、max_steps={args.max_steps} に固定",
    "negative_terms": negative_terms,
    "negative_answer": NEGATIVE_ANSWER,
    "negative_ratio": neg_ratio,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "train_examples": len(train_raw),
    "global_step": trainer.state.global_step,
    "epochs": round(trainer.state.epoch, 2),
    "trainable_params": trainable,
    "all_params": all_params,
    "trainable_pct": round(100 * trainable / all_params, 4),
    "train_seconds": round(train_seconds, 1),
    "final_loss": round(train_result.training_loss, 4),
    "before": before_summary,
    "after": after_summary,
    "changed": changed,
    "executed_training_myself": True,
}
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("=== DONE ===", WORK / "result.json")
