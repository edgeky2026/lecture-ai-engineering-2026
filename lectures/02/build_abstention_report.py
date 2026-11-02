"""コミット済みの eval_*.json から「誤拒否率」を後付けで集計する（学習の再実行は不要）。

run_glossary_lora.py の評価は 3 バケット（known / paraphrase / unknown）だけで、
**用語集にある語なのに「記載がありません」と拒否した**ケース（= problem.md がいう
「根拠ありの質問に確認要求で逃げた」型）を数える指標が無かった。
否定例を学習データに混ぜる対策（v2）はこの型を増やしうるので、対称な指標として追加する。

誤拒否率 = (known + paraphrase の 20 問のうち、HONEST_MARKERS を含む応答の数) / 20

実行: python lectures/02/build_abstention_report.py
出力: lectures/02/abstention_report.json と標準出力の表
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# run_glossary_lora.py の HONEST_MARKERS と同一（変えると指標の定義が変わるのでコピーを置く）
HONEST_MARKERS = [
    "わかりません", "わからない", "不明", "情報がありません", "記載がありません",
    "知りません", "確認が必要", "用語集に", "回答できません", "見つかりません", "定義されていません",
]

RUNS = {
    "v1（否定例なし）": "qlora_lab_glossary",
    "v2（否定例16件）": "qlora_lab_glossary_v2_unknown_aware",
}


def abstained(raw: str) -> bool:
    return any(m in raw for m in HONEST_MARKERS)


report = {"definition": "誤拒否 = 用語集に項目がある語の質問に対し、HONEST_MARKERS を含む応答を返したもの", "runs": {}}
print(f'{"run":22} {"label":7} {"誤拒否":>6} {"率":>7}  内訳')
for name, d in RUNS.items():
    report["runs"][name] = {}
    for label in ["before", "after"]:
        rows = json.loads((HERE / d / f"eval_{label}.json").read_text(encoding="utf-8"))["rows"]
        grounded = [r for r in rows if r["bucket"] in ("known", "paraphrase")]
        bad = [r for r in grounded if abstained(r["raw"])]
        rate = round(len(bad) / len(grounded), 3)
        report["runs"][name][label] = {
            "n_grounded": len(grounded), "n_false_abstention": len(bad), "rate": rate,
            "cases": [{"bucket": r["bucket"], "term_id": r["term_id"], "question": r["question"], "raw": r["raw"]} for r in bad],
        }
        print(f'{name:22} {label:7} {len(bad):>4}/{len(grounded)} {rate:>7}  '
              f'{[(r["bucket"], r["term_id"]) for r in bad]}')

(HERE / "abstention_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n書き出し:", HERE / "abstention_report.json")
