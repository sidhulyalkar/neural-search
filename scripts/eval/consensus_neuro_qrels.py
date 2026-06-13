"""Build consensus from multiple neuro_judge judgment files.

Accepts one or more --inputs files (e.g. from different backends).
Groups all judgments by (query_id, dataset_id), applies consensus rules,
and writes:
  - consensus JSONL (agreed pairs)
  - conflicts JSONL (disagreements)

Usage::

    # Single file
    python scripts/eval/consensus_neuro_qrels.py \
        --inputs artifacts/field_state/neuro_qrels_judgments_mock.jsonl \
        --out artifacts/field_state/neuro_qrels_consensus_mock.jsonl \
        --conflicts artifacts/field_state/neuro_qrels_conflicts_mock.jsonl

    # Multi-file (cross-backend consensus)
    python scripts/eval/consensus_neuro_qrels.py \
        --inputs \
          artifacts/field_state/neuro_qrels_judgments_full_openai.jsonl \
          artifacts/field_state/neuro_qrels_judgments_full_anthropic.jsonl \
        --out artifacts/field_state/neuro_qrels_consensus.jsonl \
        --conflicts artifacts/field_state/neuro_qrels_conflicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.consensus import build_consensus
from neural_search.eval.neuro_judge.evidence_packet import (
    NEURO_JUDGE_WATERMARK,
    NeuroJudgment,
)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build consensus from neuro_judge judgments")
    parser.add_argument(
        "--inputs", "--judgments",
        dest="inputs",
        nargs="+",
        default=["artifacts/field_state/neuro_qrels_judgments.jsonl"],
    )
    parser.add_argument(
        "--out", "--consensus",
        dest="out",
        default="artifacts/field_state/neuro_qrels_consensus.jsonl",
    )
    parser.add_argument(
        "--conflicts",
        default="artifacts/field_state/neuro_qrels_conflicts.jsonl",
    )
    args = parser.parse_args(argv)

    all_judgments: list[NeuroJudgment] = []
    for inp in args.inputs:
        p = _REPO / inp
        if not p.exists():
            print(f"[WARN] Input file not found (skipping): {p}", file=sys.stderr)
            continue
        print(f"Loading judgments from {p}...")
        for raw in _load_jsonl(p):
            try:
                all_judgments.append(NeuroJudgment.model_validate(raw))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] skipping malformed judgment: {exc}", file=sys.stderr)

    if not all_judgments:
        sys.exit("[ERROR] No valid judgments found in any input file.")

    # Group by (query_id, dataset_id)
    groups: dict[tuple[str, str], list[NeuroJudgment]] = defaultdict(list)
    for j in all_judgments:
        groups[(j.query_id, j.dataset_id)].append(j)

    print(
        f"Building consensus for {len(groups)} pairs from {len(all_judgments)} judgments "
        f"({len(args.inputs)} input file(s))..."
    )
    print(f"\n{NEURO_JUDGE_WATERMARK}\n")

    consensus_records: list[dict] = []
    conflict_records: list[dict] = []

    for (_qid, _did), group in groups.items():
        consensus, conflict = build_consensus(group)
        if consensus is not None:
            consensus_records.append(consensus.model_dump(mode="json"))
        if conflict is not None:
            conflict_records.append(conflict.model_dump(mode="json"))

    out_path = _REPO / args.out
    conf_path = _REPO / args.conflicts
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fh:
        for r in consensus_records:
            fh.write(json.dumps(r) + "\n")

    with conf_path.open("w") as fh:
        for r in conflict_records:
            fh.write(json.dumps(r) + "\n")

    n_total = len(groups)
    n_consensus = len(consensus_records)
    n_conflict = len(conflict_records)
    print(f"Consensus: {n_consensus}/{n_total} pairs ({100*n_consensus//max(n_total,1)}%)  → {out_path}")
    print(f"Conflicts: {n_conflict}/{n_total} pairs ({100*n_conflict//max(n_total,1)}%)   → {conf_path}")

    return {
        "total_pairs": n_total,
        "consensus": n_consensus,
        "conflicts": n_conflict,
        "consensus_rate": n_consensus / max(n_total, 1),
    }


if __name__ == "__main__":
    main()
