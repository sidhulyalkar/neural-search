#!/usr/bin/env python3
"""Update learning signals from feedback, neuro-judge judgments, and benchmark gaps.

Usage::

    python scripts/field_state/update_learning_signals.py \\
        --feedback artifacts/frontend/retrieval_feedback.jsonl \\
        --judgments artifacts/field_state/neuro_qrels_consensus.jsonl \\
        --gaps artifacts/field_state/benchmark_gaps.jsonl \\
        --out-dir artifacts/field_state \\
        --report-dir reports/field_state

Outputs:
    artifacts/field_state/learning_signals.jsonl
    reports/field_state/learning_signals.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.field_state.learning_signals import LearningSignalGenerator
from neural_search.field_state.memory_graph import _load_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("update_learning_signals")

DEFAULT_FEEDBACK = Path("artifacts/frontend/retrieval_feedback.jsonl")
DEFAULT_JUDGMENTS = Path("artifacts/field_state/neuro_qrels_consensus.jsonl")
DEFAULT_GAPS = Path("artifacts/field_state/benchmark_gaps.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/field_state")
DEFAULT_REPORT_DIR = Path("reports/field_state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    p.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    p.add_argument("--gaps", type=Path, default=DEFAULT_GAPS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen = LearningSignalGenerator()

    # Feedback
    feedback_records: list[dict] = []
    if args.feedback.exists():
        feedback_records = _load_jsonl(args.feedback)
        log.info("Loaded %d feedback records", len(feedback_records))
        gen.process_feedback(feedback_records)
    else:
        log.info("No feedback file found at %s", args.feedback)

    # Judgments
    judgments: list[dict] = []
    if args.judgments.exists():
        judgments = _load_jsonl(args.judgments)
        log.info("Loaded %d judgment records", len(judgments))
        gen.process_judgments(judgments, feedback_records=feedback_records)
    else:
        log.info("No judgments file found at %s", args.judgments)

    # Benchmark gaps
    if args.gaps.exists():
        gaps = _load_jsonl(args.gaps)
        log.info("Loaded %d benchmark gap records", len(gaps))
        gen.process_retrieval_gaps(gaps)
    else:
        log.info("No gaps file found at %s", args.gaps)

    log.info("Total signals generated: %d", len(gen.signals))

    # Write signals JSONL
    signals_path = args.out_dir / "learning_signals.jsonl"
    count = gen.export_signals(signals_path)
    log.info("Wrote %d signals → %s", count, signals_path)

    # Write audit priority queue
    audit_queue = gen.compute_audit_priority_queue()
    audit_path = args.out_dir / "expert_audit_priority_queue.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as fh:
        for entry in audit_queue:
            fh.write(json.dumps(entry, default=str) + "\n")
    log.info("Wrote audit queue → %s", audit_path)

    # Write report
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "learning_signals.md"
    report_path.write_text(gen.render_report(), encoding="utf-8")
    log.info("Wrote report → %s", report_path)

    print(f"\nLearning signals updated: {count} signals written")
    print(f"  Audit targets: {len(audit_queue)}")
    print(f"  Outputs: {signals_path}, {report_path}")


if __name__ == "__main__":
    main()
