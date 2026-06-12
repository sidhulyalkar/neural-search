"""Convert neuro_judge consensus records to QrelsEntryV1-compatible format.

This allows neuro_judge labels to be fed into the existing gold-qrels metric
reporter. The output is watermarked with the neuro_judge provenance.

Usage::

    python scripts/eval/convert_neuro_consensus_to_qrels.py \
        --input artifacts/field_state/neuro_qrels_consensus.jsonl \
        --out artifacts/field_state/neuro_qrels_consensus_for_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

NEURO_JUDGE_WATERMARK = (
    "PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD"
)

_LABEL_NAMES = {
    0: "not_relevant",
    1: "weakly_relevant",
    2: "partially_relevant",
    3: "highly_relevant",
}


def _convert(rec: dict) -> dict:
    label = int(rec.get("label", 0))
    return {
        "query_id": rec.get("query_id", ""),
        "dataset_id": rec.get("dataset_id", ""),
        "relevance": label,
        "label": _LABEL_NAMES.get(label, "unknown"),
        "rationale": rec.get("rationale_short", ""),
        "hard_negative_violation": bool(rec.get("hard_negative_detected", False)),
        "missing_metadata": list(rec.get("missing_information", [])),
        "annotator_id": f"neuro_judge:{rec.get('judge_models', ['unknown'])[0] if rec.get('judge_models') else 'unknown'}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adjudicated": False,
        "adjudication_notes": NEURO_JUDGE_WATERMARK,
        "label_provenance": rec.get("label_provenance", "neuro_judge_consensus"),
        "schema_version": "0.3",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert neuro consensus to QrelsEntryV1 format")
    parser.add_argument(
        "--input",
        default="artifacts/field_state/neuro_qrels_consensus.jsonl",
    )
    parser.add_argument(
        "--out",
        default="artifacts/field_state/neuro_qrels_consensus_for_metrics.jsonl",
    )
    args = parser.parse_args(argv)

    in_path = _REPO / args.input
    out_path = _REPO / args.out

    if not in_path.exists():
        sys.exit(f"[ERROR] Input not found: {in_path}")

    records = []
    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(_convert(rec)) + "\n")

    print(f"Converted {len(records)} records → {out_path}")
    print(f"Note: {NEURO_JUDGE_WATERMARK}")


if __name__ == "__main__":
    main()
