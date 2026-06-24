#!/usr/bin/env python3
"""Parallel LLM qrels labeling — OpenRouter, Ollama, or LM Studio.

Discovers workers from environment variables in priority order:

  1. OLLAMA_BASE_URL + OLLAMA_MODEL      → local Ollama (M1 Mac, 1 worker)
  2. LM_STUDIO_BASE_URL + LM_STUDIO_MODEL → local LM Studio (1 worker)
  3. GROQ_API_KEY                        → Groq free tier (~400 tok/s, fast!)
  4. OPENROUTER_API_KEY                  → OpenRouter (requires credits)
  5. KEYWAY_BASE_URL + legacy keys       → keyway proxy (legacy fallback)

Recommended setup for Apple Silicon (M1/M2/M3):
  # Install Ollama: https://ollama.ai
  # Pull a model:
  #   ollama pull qwen2.5:14b-instruct    # 16 GB RAM
  #   ollama pull qwen2.5:32b-instruct    # 32 GB RAM
  #   ollama pull llama3.3:70b-instruct   # 64 GB RAM
  # Then set in .env.local:
  #   OLLAMA_BASE_URL=http://localhost:11434/v1
  #   OLLAMA_MODEL=qwen2.5:14b-instruct
  #   OLLAMA_WORKERS=1   (1-2 for local inference)

All workers use the OpenAI-compatible interface so one worker class handles
all backends. Fully resumable — skips already-judged pairs on restart.

Output
------
  data/qrels/llm_judgments.jsonl   — NeuroJudgment JSONL (resumable)
  data/qrels/qrels.trec            — TREC qrels (query_id 0 record_id grade)

Usage
-----
    # Dry-run: print config, don't call APIs
    PYTHONPATH=. python scripts/eval/run_parallel_llm_qrels.py --dry-run

    # Full run (auto-detects backend from .env.local)
    PYTHONPATH=. /home/sid21/anaconda3/bin/python3 scripts/eval/run_parallel_llm_qrels.py

    # Smoke test — 5 packets
    PYTHONPATH=. /home/sid21/anaconda3/bin/python3 scripts/eval/run_parallel_llm_qrels.py --limit 5

    # Keep prior error entries instead of re-judging them
    PYTHONPATH=. /home/sid21/anaconda3/bin/python3 scripts/eval/run_parallel_llm_qrels.py --keep-errors
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# Load .env.local before importing anything that reads env vars
_ENV_LOCAL = _REPO / ".env.local"
if _ENV_LOCAL.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_LOCAL, override=False)

from neural_search.eval.neuro_judge.evidence_packet import (  # noqa: E402
    EvidencePacket,
    NeuroJudgment,
)
from neural_search.eval.neuro_judge.judge import (  # noqa: E402,PLC2701
    _error_judgment,
    _parse_judgment,
)
from neural_search.eval.neuro_judge.prompt import build_judge_prompt  # noqa: E402

from scripts.eval.docid import normalize_docid  # noqa: E402

DEFAULT_PACKETS = Path("artifacts/ablation_judge/evidence_packets.jsonl")
DEFAULT_OUT = Path("data/qrels/llm_judgments.jsonl")
DEFAULT_TREC = Path("data/qrels/qrels.trec")
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0  # seconds, doubles on each retry

# ---------------------------------------------------------------------------
# Local inference (Ollama / LM Studio) — priority 1 & 2
# ---------------------------------------------------------------------------
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")
_OLLAMA_N_WORKERS = int(os.environ.get("OLLAMA_WORKERS", "1"))

_LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "")
_LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")
_LM_STUDIO_N_WORKERS = int(os.environ.get("LM_STUDIO_WORKERS", "1"))

# ---------------------------------------------------------------------------
# Groq config — priority 3 (free tier: 7K req/day, ~400 tok/s, very fast)
# Get a free key at https://console.groq.com
# ---------------------------------------------------------------------------
_GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_GROQ_N_WORKERS = int(os.environ.get("GROQ_WORKERS", "4"))

# Groq free tier: 30 RPM / 131K TPM per model. With 4 workers and ~1500 tokens
# per request, we stay well within limits. The 7K req/day cap means ~13K pairs
# takes ~2 days (7K day 1, 6.6K day 2). Run is resumable.

# ---------------------------------------------------------------------------
# OpenRouter config — priority 4
# ---------------------------------------------------------------------------
_OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
_OPENROUTER_N_WORKERS = int(os.environ.get("OPENROUTER_WORKERS", "4"))

# ---------------------------------------------------------------------------
# Legacy keyway config — priority 5
# ---------------------------------------------------------------------------
_KEYWAY_BASE_URL = os.environ.get("KEYWAY_BASE_URL", "")

OPUS_MODEL = os.environ.get("OPUS_MODEL_NAME", "anthropic/claude-opus-4-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL_NAME", "google/gemini-2.5-flash")

_OPUS_KEY_NAMES = [
    "CLAUDE_OPUS_API_KEY",
    "CLAUDE_OPUS_API_KEY1",
    "CLAUDE_OPUS_API_KEY2",
    "CLAUDE_OPUS_API_KEY3",
    "CLAUDE_OPUS_API_KEY4",
    "CLAUDE_OPUS_API_KEY5",
]
_GEMINI_KEY_NAMES = [
    "GEMINI_API_KEY1",
    "GEMINI_API_KEY2",
    "GEMINI_API_KEY3",
    "GEMINI_API_KEY4",
    "GEMINI_API_KEY5",
    "GEMINI_API_KEY6",
]


# ---------------------------------------------------------------------------
# Worker config
# ---------------------------------------------------------------------------

class WorkerConfig:
    def __init__(self, worker_id: str, api_key: str, base_url: str, model: str) -> None:
        self.worker_id = worker_id
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}..." if self.api_key else "(no key)"
        return f"Worker({self.worker_id}, model={self.model}, key={masked})"


def _discover_workers(base_url: str = "") -> list[WorkerConfig]:  # noqa: C901
    """Discover workers in priority order: Ollama → LM Studio → OpenRouter → keyway."""
    workers: list[WorkerConfig] = []

    # Priority 1: Ollama (local, free, Apple Silicon optimized)
    ollama_url = base_url or _OLLAMA_BASE_URL
    if ollama_url and "11434" in ollama_url:  # explicit Ollama port check
        for i in range(_OLLAMA_N_WORKERS):
            workers.append(WorkerConfig(f"ollama-{i}", "ollama", ollama_url, _OLLAMA_MODEL))
        return workers

    # Priority 2: LM Studio (local, free)
    lms_url = base_url or _LM_STUDIO_BASE_URL
    if lms_url and "1234" in lms_url:
        for i in range(_LM_STUDIO_N_WORKERS):
            workers.append(WorkerConfig(f"lms-{i}", "lm-studio", lms_url, _LM_STUDIO_MODEL))
        return workers

    # Priority 3: Groq (cloud, free tier, very fast ~400 tok/s)
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        groq_url = base_url or _GROQ_BASE_URL
        for i in range(_GROQ_N_WORKERS):
            workers.append(WorkerConfig(f"groq-{i}", groq_key, groq_url, _GROQ_MODEL))
        return workers

    # Priority 4: OpenRouter (cloud, requires credits)
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if or_key:
        or_url = base_url or _OPENROUTER_BASE_URL
        for i in range(_OPENROUTER_N_WORKERS):
            workers.append(WorkerConfig(f"or-{i}", or_key, or_url, _OPENROUTER_MODEL))
        return workers

    # Priority 5: Legacy keyway proxy
    effective_url = base_url or _KEYWAY_BASE_URL
    if not effective_url:
        return workers
    for name in _OPUS_KEY_NAMES:
        key = os.environ.get(name, "").strip()
        if key:
            idx = len([w for w in workers if "opus" in w.worker_id])
            workers.append(WorkerConfig(f"opus-{idx}", key, effective_url, OPUS_MODEL))
    for name in _GEMINI_KEY_NAMES:
        key = os.environ.get(name, "").strip()
        if key:
            idx = len([w for w in workers if "gemini" in w.worker_id])
            workers.append(WorkerConfig(f"gemini-{idx}", key, effective_url, GEMINI_MODEL))

    return workers


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_already_judged(path: Path, skip_errors: bool = True) -> set[tuple[str, str]]:
    """Return set of (query_id, dataset_id) pairs that are already judged.

    When skip_errors=True (default), error entries (abstain_recommended=True with
    a judge_error rationale) are excluded so they get re-judged on the next run.
    """
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if skip_errors and "judge_error" in str(rec.get("rationale_short", "")):
                continue  # re-judge errors
            seen.add((str(rec["query_id"]), str(rec["dataset_id"])))
        except Exception:  # noqa: BLE001
            pass
    return seen


def _load_packets(path: Path, limit: int | None = None) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            packets.append(EvidencePacket.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: skipping malformed packet: {exc}", flush=True)
        if limit and len(packets) >= limit:
            break
    return packets


# ---------------------------------------------------------------------------
# Per-worker judge function
# ---------------------------------------------------------------------------

def _judge_packet(
    client: Any,
    packet: EvidencePacket,
    model: str,
    max_retries: int,
    retry_delay: float,
) -> NeuroJudgment:
    prompt = build_judge_prompt(packet)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            return _parse_judgment(text, packet, model, "v1")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = retry_delay * (2 ** attempt)
            print(
                f"    [{packet.query_id}/{packet.dataset_id}] attempt {attempt + 1} failed: "
                f"{exc!s:.80} — retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)

    return _error_judgment(packet, f"all_retries_failed: {last_exc}", model, "v1")


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(
        self,
        config: WorkerConfig,
        work_queue: queue.Queue[EvidencePacket | None],
        out_lock: threading.Lock,
        out_path: Path,
        stats: dict[str, Any],
        stats_lock: threading.Lock,
        max_retries: int,
        retry_delay: float,
    ) -> None:
        super().__init__(name=config.worker_id, daemon=True)
        self._config = config
        self._queue = work_queue
        self._out_lock = out_lock
        self._out_path = out_path
        self._stats = stats
        self._stats_lock = stats_lock
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client: Any = None

    def _init_client(self) -> None:
        import openai
        self._client = openai.OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )

    def run(self) -> None:
        self._init_client()
        while True:
            try:
                packet = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if packet is None:  # poison pill
                self._queue.task_done()
                break
            try:
                judgment = _judge_packet(
                    self._client,
                    packet,
                    self._config.model,
                    self._max_retries,
                    self._retry_delay,
                )
                self._write(judgment)
                with self._stats_lock:
                    self._stats["done"] += 1
                    if judgment.label >= 0:
                        self._stats["labeled"] += 1
                    if "error" in judgment.rationale_short.lower():
                        self._stats["errors"] += 1
            except Exception as exc:  # noqa: BLE001
                with self._stats_lock:
                    self._stats["errors"] += 1
                print(f"  [{self.name}] unexpected error: {exc}", flush=True)
            finally:
                self._queue.task_done()

    def _write(self, judgment: NeuroJudgment) -> None:
        line = judgment.model_dump_json() + "\n"
        with self._out_lock:
            with self._out_path.open("a", encoding="utf-8") as f:
                f.write(line)


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

def _progress_reporter(
    stats: dict[str, Any],
    stats_lock: threading.Lock,
    stop_event: threading.Event,
    total: int,
    interval: float = 10.0,
) -> None:
    t0 = time.time()
    while not stop_event.wait(interval):
        with stats_lock:
            done = stats["done"]
            errors = stats["errors"]
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        pct = 100 * done / total if total > 0 else 0
        eta = f"{remaining / 60:.1f}m" if remaining < 3600 else "∞"
        print(
            f"  Progress: {done}/{total} ({pct:.1f}%) | "
            f"{rate:.1f}/s | ETA {eta} | errors={errors}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# TREC qrels writer
# ---------------------------------------------------------------------------

def _write_trec_qrels(judgments_path: Path, trec_path: Path) -> int:
    """Convert judgments JSONL → TREC qrels format, excluding error entries."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in judgments_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "judge_error" in str(rec.get("rationale_short", "")):
            continue  # skip error entries — label=0 here is meaningless
        qid = str(rec["query_id"])
        did = str(rec["dataset_id"])
        label = int(rec.get("label", -1))
        if label < 0:
            continue
        key = (qid, did)
        if key in seen:
            continue
        seen.add(key)
        # TREC is whitespace-delimited: normalize doc-id to a single token so
        # ids with internal spaces don't shift columns (see scripts/eval/docid).
        lines.append(f"{qid} 0 {normalize_docid(did)} {label}\n")
    trec_path.parent.mkdir(parents=True, exist_ok=True)
    trec_path.write_text("".join(lines), encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel LLM qrels labeling.")
    parser.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trec", type=Path, default=DEFAULT_TREC)
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: all discovered keys)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N packets (for testing)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument("--base-url", type=str, default="",
                        help="Override API base URL (reads OPENROUTER_BASE_URL by default)")
    parser.add_argument("--keep-errors", action="store_true",
                        help="Treat prior judge_error entries as done (default: re-judge them)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without calling APIs")
    args = parser.parse_args(argv)

    # -- Discover workers --------------------------------------------------
    workers = _discover_workers(base_url=args.base_url)
    if args.workers:
        workers = workers[: args.workers]

    if not workers:
        raise SystemExit(
            "\nNo LLM backend configured. Add ONE of these to .env.local:\n\n"
            "  # Groq (free tier, fast — RECOMMENDED)\n"
            "  # Get key: https://console.groq.com (free, instant)\n"
            "  GROQ_API_KEY=gsk_...\n"
            "  GROQ_MODEL=llama-3.3-70b-versatile   # default\n"
            "  GROQ_WORKERS=4\n\n"
            "  # Ollama (local, free — good for M1 Mac)\n"
            "  OLLAMA_BASE_URL=http://localhost:11434/v1\n"
            "  OLLAMA_MODEL=qwen2.5:14b-instruct   # 14b is faster than 32b\n\n"
            "  # LM Studio (local, free)\n"
            "  LM_STUDIO_BASE_URL=http://localhost:1234/v1\n"
            "  LM_STUDIO_MODEL=<model-name-from-lm-studio>\n\n"
            "  # OpenRouter (cloud, requires credits)\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n"
        )

    print(f"\n{'='*60}")
    print("Parallel LLM Qrels Labeling")
    print(f"{'='*60}")
    print(f"  Packets:  {args.packets}")
    print(f"  Output:   {args.out}")
    print(f"  Workers:  {len(workers)}")
    for w in workers:
        print(f"    {w}")
    print(f"{'='*60}\n")

    if not workers:
        raise SystemExit(
            "No API keys found. Set CLAUDE_OPUS_API_KEY[1-5] and/or "
            "GEMINI_API_KEY[1-6] in environment or .env.local"
        )

    if args.dry_run:
        print("Dry-run mode — exiting without calling APIs.")
        return 0

    if not args.packets.exists():
        raise SystemExit(
            f"Evidence packets not found: {args.packets}\n"
            "Run build_ablation_evidence_packets.py first."
        )

    # -- Load packets -------------------------------------------------------
    print("Loading evidence packets ...", flush=True)
    all_packets = _load_packets(args.packets, limit=args.limit)
    print(f"  {len(all_packets)} packets loaded")

    # -- Resume: skip already-judged pairs ----------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    already_judged = _load_already_judged(args.out, skip_errors=not args.keep_errors)
    pending = [
        p for p in all_packets
        if (p.query_id, p.dataset_id) not in already_judged
    ]
    skipped = len(all_packets) - len(pending)
    if skipped:
        print(f"  Resuming: {skipped} already judged, {len(pending)} remaining")
    print(f"  Total to judge: {len(pending)}\n", flush=True)

    if not pending:
        print("Nothing to do — all packets already judged.")
        n = _write_trec_qrels(args.out, args.trec)
        print(f"TREC qrels written: {n} lines → {args.trec}")
        return 0

    # -- Build work queue ---------------------------------------------------
    work_queue: queue.Queue[EvidencePacket | None] = queue.Queue()
    for p in pending:
        work_queue.put(p)
    # Poison pills — one per worker
    for _ in workers:
        work_queue.put(None)

    # -- Shared state -------------------------------------------------------
    out_lock = threading.Lock()
    stats_lock = threading.Lock()
    stats: dict[str, Any] = {"done": 0, "labeled": 0, "errors": 0}
    stop_reporter = threading.Event()

    # -- Start workers -------------------------------------------------------
    threads = [
        WorkerThread(
            config=cfg,
            work_queue=work_queue,
            out_lock=out_lock,
            out_path=args.out,
            stats=stats,
            stats_lock=stats_lock,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        for cfg in workers
    ]

    reporter = threading.Thread(
        target=_progress_reporter,
        args=(stats, stats_lock, stop_reporter, len(pending)),
        daemon=True,
    )

    print(f"Starting {len(threads)} worker threads ...\n", flush=True)
    t0 = time.time()
    reporter.start()
    for t in threads:
        t.start()

    # -- Wait for completion ------------------------------------------------
    work_queue.join()
    stop_reporter.set()
    reporter.join(timeout=2.0)

    elapsed = time.time() - t0
    with stats_lock:
        done = stats["done"]
        errors = stats["errors"]

    print(f"\n{'='*60}")
    print(f"Labeling complete: {done} judgments in {elapsed:.1f}s")
    print(f"  Rate: {done / elapsed:.1f} judgments/s | Errors: {errors}")
    print(f"{'='*60}\n")

    # -- Write TREC qrels ---------------------------------------------------
    n = _write_trec_qrels(args.out, args.trec)
    print(f"TREC qrels: {n} lines → {args.trec}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
