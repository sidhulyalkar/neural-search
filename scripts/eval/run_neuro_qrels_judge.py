"""Run the neuro_judge LLM over evidence packets and write judgment records.

Usage::

    # Mock backend (no API keys needed)
    python scripts/eval/run_neuro_qrels_judge.py \
        --packets artifacts/field_state/neuro_judge_evidence_packets.jsonl \
        --out artifacts/field_state/neuro_qrels_judgments_mock.jsonl \
        --backend mock --limit 50 --resume

    # Anthropic backend
    python scripts/eval/run_neuro_qrels_judge.py \
        --packets artifacts/field_state/neuro_judge_evidence_packets.jsonl \
        --out artifacts/field_state/neuro_qrels_judgments_sample_anthropic.jsonl \
        --backend anthropic --model claude-haiku-4-5-20251001 \
        --limit 50 --temperature 0 --resume
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.evidence_packet import (  # noqa: E402
    NEURO_JUDGE_WATERMARK,
    EvidencePacket,
    NeuroJudgment,
)
from neural_search.eval.neuro_judge.judge import build_neuro_judge  # noqa: E402
from neural_search.eval.neuro_judge.prompt import build_judge_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_already_judged(path: Path) -> set[tuple[str, str]]:
    """Return set of (query_id, dataset_id) already in the output file."""
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for rec in _load_jsonl(path):
        qid = str(rec.get("query_id", ""))
        did = str(rec.get("dataset_id", ""))
        if qid and did:
            seen.add((qid, did))
    return seen


def _load_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for rec in _load_jsonl(path):
        cache_key = str(rec.get("_cache_key") or "")
        if cache_key:
            cache[cache_key] = rec
    return cache


def _append_cache(path: Path | None, records: list[dict]) -> None:
    if path is None or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _cache_key(packet: EvidencePacket, backend: str, model: str | None, prompt_version: str | None) -> str:
    return "|".join([backend, model or "", prompt_version or "", packet.packet_hash()])


def _estimate_tokens(packets: list[EvidencePacket], n_judges: int) -> dict[str, int]:
    prompt_chars = sum(len(build_judge_prompt(packet)) for packet in packets) * n_judges
    return {
        "packets": len(packets),
        "judge_invocations": len(packets) * n_judges,
        "estimated_prompt_tokens": max(1, prompt_chars // 4),
    }


def _run_judges(
    packets: list[EvidencePacket],
    backend: str,
    model: str | None,
    n_judges: int,
    temperature: float,
    prompt_version: str | None,
    cache_path: Path | None = None,
) -> tuple[list[dict], int]:
    """Run n_judges instances on all packets. Returns (judgments, error_count)."""
    kwargs: dict = {}
    if model:
        kwargs["model"] = model
    judges = [build_neuro_judge(backend, **kwargs) for _ in range(n_judges)]
    judgments: list[dict] = []
    errors = 0
    cache = _load_cache(cache_path)
    new_cache_records: list[dict] = []

    for i, packet in enumerate(packets, 1):
        for judge in judges:
            try:
                key = _cache_key(packet, backend, model or judge.model_id, prompt_version)
                cached = cache.get(key)
                if cached:
                    cached_payload = {k: v for k, v in cached.items() if k != "_cache_key"}
                    j = NeuroJudgment.model_validate(cached_payload)
                else:
                    j = judge.judge(packet)
                d = j.model_dump(mode="json")
                if temperature != 0:
                    d["_temperature"] = temperature
                if prompt_version:
                    d["prompt_version"] = prompt_version
                judgments.append(d)
                if not cached:
                    cache_record = dict(d)
                    cache_record["_cache_key"] = key
                    new_cache_records.append(cache_record)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(
                    f"[WARN] judge error for {packet.query_id}/{packet.dataset_id}: {exc}",
                    file=sys.stderr,
                )
        if i % 100 == 0 or i == len(packets):
            print(f"  judged {i}/{len(packets)} packets...", file=sys.stderr)
    _append_cache(cache_path, new_cache_records)
    return judgments, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run neuro_judge over evidence packets")
    parser.add_argument(
        "--packets", "--evidence",
        dest="packets",
        default="artifacts/field_state/neuro_judge_evidence_packets.jsonl",
    )
    parser.add_argument(
        "--out", "--output",
        dest="output",
        default="artifacts/field_state/neuro_qrels_judgments.jsonl",
    )
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=["mock", "anthropic", "openai", "local_hf", "braingpt"],
    )
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument(
        "--judges",
        type=int,
        default=1,
        help="Number of independent judge instances",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N packets",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip packets already present in the output file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Alias for --resume",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Write and validate output after every N packets",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional JSONL cache path keyed by backend/model/prompt/evidence hash",
    )
    parser.add_argument(
        "--dry-run-cost-estimate",
        action="store_true",
        help="Estimate prompt tokens and exit without judging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (informational; passed through to judgment metadata)",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Override prompt version tag stored in judgments",
    )
    args = parser.parse_args(argv)

    packets_path = _REPO / args.packets
    out_path = _REPO / args.output
    cache_path = (_REPO / args.cache) if args.cache else None

    if not packets_path.exists():
        sys.exit(f"[ERROR] Packets file not found: {packets_path}")

    print(f"Loading evidence packets from {packets_path}...")
    raw_packets = _load_jsonl(packets_path)
    packets = [EvidencePacket.model_validate(r) for r in raw_packets]

    # Resume: skip already-judged pairs
    already_judged: set[tuple[str, str]] = set()
    resume = bool(args.resume or args.skip_existing)
    if resume:
        already_judged = _load_already_judged(out_path)
        before = len(packets)
        packets = [p for p in packets if (p.query_id, p.dataset_id) not in already_judged]
        if already_judged:
            print(f"  Resuming: skipping {before - len(packets)} already-judged packets.")

    # Limit
    if args.limit is not None:
        packets = packets[: args.limit]

    if args.dry_run_cost_estimate:
        print(json.dumps(_estimate_tokens(packets, args.judges), indent=2))
        return {"estimated": True, **_estimate_tokens(packets, args.judges)}

    print(f"Running {args.judges} judge(s) via {args.backend} on {len(packets)} packets...")
    print(f"\n{NEURO_JUDGE_WATERMARK}\n")

    # Append mode when resuming, otherwise write fresh
    write_mode = "a" if resume and already_judged else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    errors = 0
    batch_size = max(1, args.batch_size)
    with out_path.open(write_mode, encoding="utf-8") as fh:
        for start in range(0, len(packets), batch_size):
            batch = packets[start : start + batch_size]
            judgments, batch_errors = _run_judges(
                batch,
                args.backend,
                args.model,
                args.judges,
                args.temperature,
                args.prompt_version,
                cache_path=cache_path,
            )
            errors += batch_errors
            for j in judgments:
                NeuroJudgment.model_validate(j)
                fh.write(json.dumps(j) + "\n")
            fh.flush()
            total_written += len(judgments)
            print(
                f"  wrote batch {start // batch_size + 1}: "
                f"{len(judgments)} judgments (total new: {total_written})",
                file=sys.stderr,
            )

    total = len(already_judged) + total_written
    print(f"Wrote {total_written} new judgments to {out_path} (total: {total})")
    if errors:
        print(f"[WARN] {errors} judgment errors encountered", file=sys.stderr)
    return {"written": total_written, "errors": errors, "skipped": len(already_judged)}


if __name__ == "__main__":
    main()
