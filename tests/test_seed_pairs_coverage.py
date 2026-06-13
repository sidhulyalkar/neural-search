# tests/test_seed_pairs_coverage.py
import json
from pathlib import Path

SEED_FILE = Path("data/eval/usefulness_seed_pairs.jsonl")


def test_seed_file_exists():
    assert SEED_FILE.exists(), "Seed JSONL file must exist"


def test_all_intents_covered():
    """Every UsefulnessIntent must have at least 2 query IDs."""
    pairs = [json.loads(line) for line in SEED_FILE.read_text().splitlines() if line.strip()]
    intent_queries: dict[str, set] = {}
    for p in pairs:
        intent = p["intent"]
        qid = p["query_id"]
        intent_queries.setdefault(intent, set()).add(qid)

    required_intents = {
        "strict_lookup", "replication", "meta_analysis",
        "pipeline_reuse", "cross_dataset_comparison",
        "exploration", "method_transfer",
    }
    for intent in required_intents:
        count = len(intent_queries.get(intent, set()))
        assert count >= 2, f"Intent '{intent}' has only {count} query(ies); need >=2"


def test_minimum_30_pairs():
    pairs = [line for line in SEED_FILE.read_text().splitlines() if line.strip()]
    assert len(pairs) >= 99
