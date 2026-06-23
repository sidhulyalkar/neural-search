#!/usr/bin/env python3
"""Derive a BM25/dense-ready eval corpus from evidence packets.

The evidence packets are the same source the qrels were built from, so a corpus
derived from them is guaranteed to cover exactly the judged datasets. One record
per unique dataset_id; fields renamed to what SparseIndex expects.

Usage:
    PYTHONPATH=. python scripts/eval/build_corpus_from_packets.py   # run from repo root
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_PACKETS = Path("artifacts/ablation_judge/evidence_packets.jsonl")
DEFAULT_OUT = Path("data/eval/ablation_corpus_from_packets.jsonl")


def build_corpus_records(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in packets:
        did = str(p.get("dataset_id", ""))
        if not did or did in by_id:
            continue
        source, _, source_id = did.partition(":")
        by_id[did] = {
            "dataset_id": did,
            "source": p.get("source_archive") or source,
            "source_id": source_id or did,
            "title": p.get("title", ""),
            "description": p.get("description", ""),
            "modalities": list(p.get("dataset_modalities", []) or []),
            "species": list(p.get("dataset_species", []) or []),
            "tasks": list(p.get("dataset_tasks", []) or []),
            "brain_regions": list(p.get("dataset_brain_regions", []) or []),
            "data_standards": list(p.get("data_standards", []) or []),
        }
    return list(by_id.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.packets.exists():
        print(f"Packets file not found: {args.packets}", file=sys.stderr)
        return 1
    packets = [json.loads(l) for l in args.packets.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = build_corpus_records(packets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} unique dataset records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
