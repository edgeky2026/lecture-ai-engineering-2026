"""自動採点（キーワード部分一致）の妥当性を、手採点ラベルと突き合わせて測る。

なぜ要るか
  run_glossary_lora.py の採点は MUST_INCLUDE のキーワードが応答に含まれるかだけを見る。
  これは**甘い**（語が入っていれば中身が誤りでも ○）と同時に**厳しい**（要点が正しくても語が
  無ければ ×）。実測すると、この物差しだけで「否定例を入れるとトレードオフが起きる」という
  結論が出てしまい、手採点するとその差は消えた。∴ 自動採点の数字は手採点と並べて出す。

何をするか
  1. handscore_labels.json（人の判断）を eval_*.json の実出力と突き合わせる
  2. 出力が採点時点から変わっていないか sha1 で確認する（変わっていたら落ちる＝ラベルの陳腐化検出）
  3. 条件ごとに「自動の率／手採点の率」と、自動の偽陽性・偽陰性の件数を出す

実行: python lectures/02/build_handscore_report.py
出力: lectures/02/handscore_report.json と標準出力の表
"""

import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LABELS = json.loads((HERE / "handscore_labels.json").read_text(encoding="utf-8"))
GLOSSARY = {t["id"]: t for t in json.loads((REPO / "data" / "glossary.json").read_text(encoding="utf-8"))["terms"]}

# run 名 → 実出力の在り処。retrieval_vs_ft.json 側は rows[<run名>] に入っている
EVAL_FILES = {
    "before": ("qlora_lab_glossary/eval_before.json", None),
    "v1_after": ("qlora_lab_glossary/eval_after.json", None),
    "v2_after": ("qlora_lab_glossary_v2_unknown_aware/eval_after.json", None),
    "v2b_after": ("qlora_lab_v2b_neg08_step300/eval_after.json", None),
    "v3_after": ("qlora_lab_v3_neg23_step300/eval_after.json", None),
    "A_検索のみ": ("retrieval_vs_ft.json", "A_検索のみ"),
    "B_学習のみ_指示あり": ("retrieval_vs_ft.json", "B_学習のみ_指示あり"),
    "C_学習と検索の併用": ("retrieval_vs_ft.json", "C_学習と検索の併用"),
}
RUN_NOTE = {
    "before": "素のベースモデル（学習なし・指示なし）",
    "v1_after": "学習のみ・否定例0%・300step",
    "v2_after": "学習のみ・否定例12.1%・340step（ステップが揃っていない）",
    "v2b_after": "学習のみ・否定例12.1%・300step（統制済み）",
    "v3_after": "学習のみ・否定例28.4%・300step（統制済み）",
    "A_検索のみ": "検索のみ・学習なし（RAG相当）",
    "B_学習のみ_指示あり": "学習のみ＋「記載が無ければそう言え」の指示",
    "C_学習と検索の併用": "学習＋検索の併用",
}


def load_rows(run: str):
    path, key = EVAL_FILES[run]
    data = json.loads((HERE / path).read_text(encoding="utf-8"))
    rows = data["rows"][key] if key else data["rows"]
    return {(r["bucket"], r["term_id"]): r for r in rows if r["bucket"] != "unknown"}


report = {"criterion": LABELS["criterion"], "labeler": LABELS["labeler"], "runs": {}}
stale = []
print(f'{"条件":28} {"バケット":11} {"自動":>6} {"手採点":>7} {"偽陽性":>6} {"偽陰性":>6} {"逐語":>6}')
for run in EVAL_FILES:
    rows = load_rows(run)
    report["runs"][run] = {"note": RUN_NOTE[run], "buckets": {}}
    for bucket in ["known", "paraphrase"]:
        sub = [(tid, r) for (b, tid), r in rows.items() if b == bucket]
        assert len(sub) == 10, f"{run}/{bucket} が10問でない: {len(sub)}"
        fp, fn, auto_ok, hand_ok, cases = 0, 0, 0, 0, []
        sims = []
        for tid, r in sub:
            lab = LABELS["labels"][f"{run}|{bucket}|{tid}"]
            # 用語集の定義文との類似度。1.00 は「定義文をそのまま出した」＝暗記であり、
            # 正答率が高いことと言い換えに答えられることは別だと示すために測る。
            sims.append(difflib.SequenceMatcher(None, GLOSSARY[tid]["definition"], r["raw"]).ratio())
            # 採点時点の出力と今のファイルが一致しているか（再実行で変われば手採点は無効）
            if hashlib.sha1(r["raw"].encode("utf-8")).hexdigest()[:8] != lab["raw_sha1_8"]:
                stale.append(f"{run}|{bucket}|{tid}")
            assert bool(r["ok"]) == lab["auto_ok"], f"自動採点がラベルと不一致: {run}|{bucket}|{tid}"
            auto_ok += lab["auto_ok"]
            hand_ok += lab["hand_ok"]
            if lab["auto_ok"] and not lab["hand_ok"]:
                fp += 1
                cases.append({"term_id": tid, "type": "偽陽性", "reason": lab["reason"]})
            if lab["hand_ok"] and not lab["auto_ok"]:
                fn += 1
                cases.append({"term_id": tid, "type": "偽陰性", "reason": lab["reason"]})
        verbatim = sum(s >= 0.99 for s in sims)
        report["runs"][run]["buckets"][bucket] = {
            "n": 10, "auto_rate": auto_ok / 10, "hand_rate": hand_ok / 10,
            "false_positive": fp, "false_negative": fn,
            "mean_similarity_to_definition": round(sum(sims) / len(sims), 3),
            "verbatim_count": verbatim, "disagreements": cases,
        }
        print(f'{run:28} {bucket:11} {auto_ok/10:>6.2f} {hand_ok/10:>7.2f} {fp:>6} {fn:>6} {verbatim:>6}')

tot_fp = sum(b["false_positive"] for r in report["runs"].values() for b in r["buckets"].values())
tot_fn = sum(b["false_negative"] for r in report["runs"].values() for b in r["buckets"].values())
n_all = 10 * 2 * len(EVAL_FILES)
report["overall"] = {"n": n_all, "false_positive": tot_fp, "false_negative": tot_fn,
                     "disagreement_rate": round((tot_fp + tot_fn) / n_all, 3)}
print(f'\n合計 {n_all} 問: 偽陽性 {tot_fp} / 偽陰性 {tot_fn} = 自動採点の不一致率 {report["overall"]["disagreement_rate"]}')

if stale:
    raise SystemExit(f"出力が手採点時点から変わっている（ラベルが無効）: {stale}")
(HERE / "handscore_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("書き出し:", HERE / "handscore_report.json")
