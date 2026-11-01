"""宿題（発展）の追加検証: 「用語集を検索する仕組み」と「学習させる仕組み」を同じ評価で比べる。

problem.md の発展レベルは「用語集を検索する仕組みと比較して、モデルを学習させる際の
メリットとデメリットを考察する」ことを求めている。run_glossary_lora.py は学習側だけを
測っていたので、同じ 28 問を検索側にも通して**同じ物差しで**並べる。

比べる 4 条件（評価の 28 問と採点は run_glossary_lora.py と同一）
  A. 検索のみ   : 素のベースモデル + 用語集から引いた抜粋を文脈に渡す（RAG 相当。学習しない）
  B. 学習のみ   : v1 アダプタ + 「記載が無ければそう言え」という指示のみ（文脈に用語集を渡さない）
  C. 学習+検索  : v1 アダプタ + 抜粋を文脈に渡す（併用）
  D. 学習のみ・指示なし : 既測。qlora_lab_glossary/eval_after.json をそのまま参照する

検索器は「最長一致の辞書引き」: 質問文に部分一致した term / alias のうち、一致した
文字列が最も長い項目を 1 件返す（2 文字未満は採らない）。当たらなければ「該当なし」。
埋め込み検索ではないので、これは検索側の**下限**の性能である。

実行: CUDA_VISIBLE_DEVICES=1 python lectures/02/run_retrieval_vs_ft.py
出力: lectures/02/retrieval_vs_ft.json
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ADAPTER = HERE / "qlora_lab_glossary" / "adapter"
OUT = HERE / "retrieval_vs_ft.json"

BASE = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_NEW_TOKENS = 96
torch.manual_seed(42)

# ---- 評価データ：run_glossary_lora.py と同一（物差しを変えないためコピー） ----------
glossary = json.loads((REPO / "data" / "glossary.json").read_text(encoding="utf-8"))["terms"]
gmap = {t["id"]: t for t in glossary}

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
    "Few-shot learning（少数ショット学習）とは何ですか？",
    "ReActパターンとは何ですか？",
]
HONEST_MARKERS = [
    "わかりません", "わからない", "不明", "情報がありません", "記載がありません",
    "知りません", "確認が必要", "用語集に", "回答できません", "見つかりません", "定義されていません",
]

QUESTIONS = []  # (bucket, question, term_id)
for tid in KNOWN_IDS:
    QUESTIONS.append(("known", f"「{gmap[tid]['term']}」とは何ですか？", tid))
for i, tid in enumerate(KNOWN_IDS):
    t = gmap[tid]
    name = t["aliases"][0] if t["aliases"] else t["term"]
    QUESTIONS.append(("paraphrase", PARAPHRASE_TEMPLATES[i % 2].format(term=name), tid))
for q in UNKNOWN_Q:
    QUESTIONS.append(("unknown", q, None))

# ---- 検索器：最長一致の辞書引き --------------------------------------------------
NO_HIT = "該当なし"


def retrieve(question: str):
    best = (0, None)  # (一致した文字列の長さ, term)
    for t in glossary:
        for surface in [t["term"], *t.get("aliases", [])]:
            s = surface.strip()
            if len(s) < 2:
                continue
            if s.lower() in question.lower() and len(s) > best[0]:
                best = (len(s), t)
    if best[1] is None:
        return NO_HIT, None, 0
    t = best[1]
    return f"- {t['term']}: {t['definition']}", t["id"], best[0]


# ---- 生成 -----------------------------------------------------------------------
SYS_RETRIEVAL = (
    "あなたは講座用語集の回答係です。与えられた用語集の抜粋だけを根拠に、1〜2文で簡潔に答えてください。"
    f"抜粋が「{NO_HIT}」のときは、推測せず「用語集に記載がありません」と答えてください。"
)
SYS_FT_ONLY = (
    "あなたは講座用語集の回答係です。講座用語集に記載のある語は1〜2文でその定義を答えてください。"
    "用語集に記載が無い語については、推測せず「用語集に記載がありません」と答えてください。"
)

tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else None)

base = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION, dtype=torch.float32).to(DEVICE)
base.config.use_cache = True
base.eval()


def generate(m, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score(bucket, tid, raw):
    if bucket == "unknown":
        return any(w in raw for w in HONEST_MARKERS)
    return any(w in raw for w in MUST_INCLUDE[tid])


def run_condition(m, label, use_retrieval, system):
    rows, t0 = [], time.time()
    for bucket, q, tid in QUESTIONS:
        if use_retrieval:
            excerpt, hit_id, hit_len = retrieve(q)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": f"用語集の抜粋:\n{excerpt}\n\n質問: {q}"}]
        else:
            excerpt, hit_id, hit_len = None, None, 0
            messages = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        raw = generate(m, messages)
        rows.append({"bucket": bucket, "question": q, "term_id": tid, "retrieved_id": hit_id,
                     "retrieved_len": hit_len, "raw": raw, "ok": score(bucket, tid, raw),
                     "abstained": any(w in raw for w in HONEST_MARKERS)})
    summary = {"label": label, "n": len(rows), "seconds": round(time.time() - t0, 1)}
    for b in ["known", "paraphrase", "unknown"]:
        sub = [r for r in rows if r["bucket"] == b]
        summary[b] = {"n": len(sub), "rate": round(sum(r["ok"] for r in sub) / len(sub), 3)}
    grounded = [r for r in rows if r["bucket"] != "unknown"]
    summary["false_abstention"] = {
        "n": len(grounded),
        "rate": round(sum(r["abstained"] for r in grounded) / len(grounded), 3),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows


# 検索器そのものの当たり外れ（モデルを通さずに確認できるので先に出す）
retrieval_trace = []
for bucket, q, tid in QUESTIONS:
    _, hit_id, hit_len = retrieve(q)
    correct = (hit_id == tid) if bucket != "unknown" else (hit_id is None)
    retrieval_trace.append({"bucket": bucket, "question": q, "gold_id": tid,
                            "retrieved_id": hit_id, "matched_chars": hit_len, "retrieval_correct": correct})
hit_rate = {
    b: round(sum(r["retrieval_correct"] for r in retrieval_trace if r["bucket"] == b)
             / len([r for r in retrieval_trace if r["bucket"] == b]), 3)
    for b in ["known", "paraphrase", "unknown"]
}
print("検索器の正答率（unknownは「当てない」が正解）:", json.dumps(hit_rate, ensure_ascii=False))

results = {}
results["A_検索のみ"] = run_condition(base, "A_検索のみ", True, SYS_RETRIEVAL)

tuned = PeftModel.from_pretrained(base, str(ADAPTER))
tuned.config.use_cache = True
tuned.eval()
results["B_学習のみ_指示あり"] = run_condition(tuned, "B_学習のみ_指示あり", False, SYS_FT_ONLY)
results["C_学習と検索の併用"] = run_condition(tuned, "C_学習と検索の併用", True, SYS_RETRIEVAL)

OUT.write_text(json.dumps({
    "note": "同じ28問・同じ採点で、検索（RAG相当）と学習（LoRA）を比べたもの。Dは qlora_lab_glossary/eval_after.json を参照",
    "retriever": "最長一致の辞書引き（term/alias の部分一致、2文字以上、最長を1件）",
    "retrieval_hit_rate": hit_rate,
    "retrieval_trace": retrieval_trace,
    "summaries": {k: v[0] for k, v in results.items()},
    "rows": {k: v[1] for k, v in results.items()},
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("=== DONE ===", OUT)
