#!/usr/bin/env python3
"""Generate auto-labeled query-candidate pairs for usefulness benchmark expansion.

Loads the normalized real corpus directly, scores datasets against each query
using fast keyword/ontology matching (no full search pipeline), assigns usefulness
labels, and appends labeled pairs to data/eval/usefulness_seed_pairs.jsonl.

This approach is much faster than calling search_datasets() since it bypasses
the semantic expansion pipeline and loads the corpus once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

ANNOTATION_QUERIES = [
    ("qa001", "mouse hippocampus place cells neuropixels", "exploration"),
    ("qa002", "replicate Steinmetz 2019 visual cortex recording", "replication"),
    ("qa003", "calcium imaging striatum reward learning", "pipeline_reuse"),
    ("qa004", "compare EEG and MEG motor cortex datasets", "cross_dataset_comparison"),
    ("qa005", "NWB formatted primate prefrontal cortex", "strict_lookup"),
    ("qa006", "meta-analysis visual cortex across species", "meta_analysis"),
    ("qa007", "transfer GLM from mouse to rat decision-making", "method_transfer"),
    ("qa008", "two-photon imaging barrel cortex sensory whisker", "exploration"),
    ("qa009", "fiber photometry dopamine reward signal", "pipeline_reuse"),
    ("qa010", "human intracranial EEG seizure detection", "strict_lookup"),
    ("qa011", "macaque V1 electrophysiology orientation tuning", "replication"),
    ("qa012", "neuropixels prefrontal cortex working memory", "strict_lookup"),
    ("qa013", "mouse motor cortex reach and grasp task", "method_transfer"),
    ("qa014", "comparing hippocampal theta oscillations across species", "cross_dataset_comparison"),
    ("qa015", "calcium imaging cerebellar Purkinje cells", "exploration"),
    ("qa016", "large-scale neural population recordings for PCA", "method_transfer"),
    ("qa017", "datasets similar to IBL brain-wide map", "pipeline_reuse"),
    ("qa018", "auditory cortex tonotopy NWB", "strict_lookup"),
    ("qa019", "datasets for Bayesian decoder model fitting", "method_transfer"),
    ("qa020", "meta-analysis dopamine reward prediction error", "meta_analysis"),
]

# Signals for auto-labeling — maps query_id to relevance signals
RELEVANCE_SIGNALS = {
    "qa001": {"positive": ["hippocampus", "place", "neuropixels", "mouse"], "negative": ["human", "fmri"]},
    "qa002": {"positive": ["visual", "cortex", "neuropixels", "mouse"], "negative": ["calcium", "zebrafish"]},
    "qa003": {"positive": ["calcium", "striatum", "reward", "dopamine"], "negative": ["neuropixels", "primate"]},
    "qa004": {"positive": ["eeg", "meg", "motor"], "negative": ["calcium", "neuropixels"]},
    "qa005": {"positive": ["nwb", "primate", "prefrontal"], "negative": ["zebrafish", "bids-only"]},
    "qa006": {"positive": ["visual", "cortex", "v1", "v4"], "negative": ["auditory", "hippocampus"]},
    "qa007": {"positive": ["decision", "mouse", "rat", "choice"], "negative": ["passive", "spontaneous"]},
    "qa008": {"positive": ["two-photon", "barrel", "cortex", "whisker", "somatosensory"], "negative": ["primate", "ecog"]},
    "qa009": {"positive": ["fiber", "photometry", "dopamine", "reward", "striatum"], "negative": ["neuropixels", "ecog"]},
    "qa010": {"positive": ["human", "intracranial", "ecog", "ieeg", "seizure"], "negative": ["mouse", "calcium"]},
    "qa011": {"positive": ["macaque", "v1", "visual", "orientation"], "negative": ["mouse", "fmri"]},
    "qa012": {"positive": ["neuropixels", "prefrontal", "working memory"], "negative": ["calcium", "human"]},
    "qa013": {"positive": ["motor", "cortex", "reach", "mouse"], "negative": ["primate", "fmri"]},
    "qa014": {"positive": ["hippocampus", "theta", "oscillation"], "negative": ["visual", "auditory"]},
    "qa015": {"positive": ["calcium", "cerebellar", "cerebellum", "purkinje"], "negative": ["cortex", "primate"]},
    "qa016": {"positive": ["population", "neural", "recording", "pca", "dimensionality"], "negative": ["single-unit", "fmri"]},
    "qa017": {"positive": ["ibl", "brain-wide", "neuropixels", "decision"], "negative": ["calcium", "human"]},
    "qa018": {"positive": ["auditory", "tonotopy", "nwb"], "negative": ["visual", "motor"]},
    "qa019": {"positive": ["behavioral", "choice", "decision", "trial"], "negative": ["passive", "resting"]},
    "qa020": {"positive": ["dopamine", "reward", "prediction", "striatum", "basal ganglia"], "negative": ["visual", "motor"]},
}

NORMALIZED_CORPUS_DIR = project_root / "data" / "corpus" / "normalized"
SEED_FILE = project_root / "data" / "eval" / "usefulness_seed_pairs.jsonl"
OUT_FILE = project_root / "data" / "eval" / "annotation_candidates.jsonl"

# Minimum number of candidates to emit per query
CANDIDATES_PER_QUERY = 5


def load_real_corpus() -> list[dict]:
    """Load all real normalized corpus records (skip demo/backup files)."""
    records: list[dict] = []
    for jsonl_file in sorted(NORMALIZED_CORPUS_DIR.glob("*.jsonl")):
        name = jsonl_file.name.lower()
        if "demo" in name or "backup" in name or "papers" in name:
            continue
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        # Skip demo datasets that may be embedded in real files
                        sid = rec.get("source_id", "") or rec.get("dataset_id", "")
                        if str(sid).startswith("DEMO_"):
                            continue
                        records.append(rec)
                    except json.JSONDecodeError:
                        pass
    return records


def _record_text(rec: dict) -> str:
    """Build a flat text blob from all useful fields in a normalized record."""
    parts: list[str] = []

    for field in ("title", "description"):
        val = rec.get(field, "")
        if val:
            parts.append(str(val))

    # Structured label fields — extract .id / .label from list-of-dicts
    for field in ("species", "modalities", "brain_regions", "tasks",
                  "behavioral_events", "analysis_goals", "data_standards",
                  "analysis_affordances"):
        for item in rec.get(field) or []:
            if isinstance(item, dict):
                parts.append(item.get("id", ""))
                parts.append(item.get("label", ""))
            elif isinstance(item, str):
                parts.append(item)

    # Usability flags as text
    flags = rec.get("usability_flags") or {}
    for k, v in flags.items():
        if v:
            parts.append(k)

    return " ".join(p for p in parts if p).lower()


def _auto_label(text: str, qid: str) -> str:
    """Assign usefulness label based on relevance signal matching against text."""
    signals = RELEVANCE_SIGNALS.get(qid, {"positive": [], "negative": []})

    neg_hits = sum(1 for term in signals["negative"] if term.lower() in text)
    if neg_hits >= 2:
        return "not_useful"

    pos_hits = sum(1 for term in signals["positive"] if term.lower() in text)
    total_pos = len(signals["positive"])

    if pos_hits == 0:
        return "not_useful"
    elif pos_hits >= max(2, total_pos // 2):
        return "highly_useful"
    elif pos_hits >= 1:
        return "useful"
    else:
        return "weakly_useful"


def _score_record(text: str, qid: str) -> int:
    """Score a record for ranking purposes (higher = more relevant)."""
    signals = RELEVANCE_SIGNALS.get(qid, {"positive": [], "negative": []})
    neg_hits = sum(1 for term in signals["negative"] if term.lower() in text)
    pos_hits = sum(1 for term in signals["positive"] if term.lower() in text)
    return pos_hits - neg_hits * 2


def main() -> None:
    # Load existing query IDs to avoid duplicates
    existing_qids: set[str] = set()
    if SEED_FILE.exists():
        for line in SEED_FILE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing_qids.add(row.get("query_id", ""))

    print(f"Existing labeled queries: {len(existing_qids)} unique query IDs in seed file")

    # Load real corpus once
    print("Loading real corpus...", flush=True)
    corpus = load_real_corpus()
    print(f"  Loaded {len(corpus)} real corpus records from {NORMALIZED_CORPUS_DIR}")

    # Pre-compute text blobs for all records
    corpus_texts = [(rec, _record_text(rec)) for rec in corpus]

    new_pairs: list[dict] = []
    for qid, query, intent in ANNOTATION_QUERIES:
        if qid in existing_qids:
            print(f"  {qid}: already in seed file, skipping")
            continue

        print(f"  {qid}: {query[:70]}", flush=True)

        # Score and rank all records for this query
        scored: list[tuple[int, dict, str]] = []
        for rec, text in corpus_texts:
            score = _score_record(text, qid)
            scored.append((score, rec, text))

        # Sort by score descending, take top candidates across label types
        scored.sort(key=lambda x: x[0], reverse=True)

        # Collect a balanced mix: top positives + some negatives
        seen_ids: set[str] = set()
        candidates: list[tuple[int, dict, str]] = []

        # Take top 5 by score (includes positives)
        for score, rec, text in scored[:CANDIDATES_PER_QUERY]:
            dataset_id = str(rec.get("dataset_id") or rec.get("source_id") or "unknown")
            if dataset_id not in seen_ids:
                seen_ids.add(dataset_id)
                candidates.append((score, rec, text))

        # Also add 2 hard negatives (score <= 0) for diversity
        neg_count = 0
        for score, rec, text in reversed(scored):
            if neg_count >= 2:
                break
            dataset_id = str(rec.get("dataset_id") or rec.get("source_id") or "unknown")
            if dataset_id not in seen_ids and score <= 0:
                seen_ids.add(dataset_id)
                candidates.append((score, rec, text))
                neg_count += 1

        for score, rec, text in candidates:
            dataset_id = str(rec.get("dataset_id") or rec.get("source_id") or "unknown")
            label = _auto_label(text, qid)
            title = rec.get("title", "")[:60]
            print(f"    [{score:+d}] {dataset_id}: {label}  ({title})")

            new_pairs.append({
                "query_id": qid,
                "query": query,
                "intent": intent,
                "candidate_id": dataset_id,
                "usefulness_label": label,
                "label_type": "auto_signal_match",
                "notes": f"Auto-labeled from real-corpus keyword match (signal_score={score}): {label}",
            })

    # Write candidates file
    OUT_FILE.parent.mkdir(exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for pair in new_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"\nGenerated {len(new_pairs)} candidate pairs -> {OUT_FILE}")

    # Append to seed file (all auto-labeled pairs)
    with SEED_FILE.open("a", encoding="utf-8") as f:
        for pair in new_pairs:
            f.write(json.dumps(pair) + "\n")

    total = len([line for line in SEED_FILE.read_text().splitlines() if line.strip()])
    print(f"Appended {len(new_pairs)} pairs -> {SEED_FILE}")
    print(f"Total seed pairs now: {total}")


if __name__ == "__main__":
    main()
