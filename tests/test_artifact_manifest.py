"""Tests for reports/eval/current_artifact_manifest.json — the canonical artifact truth file."""
import json
from pathlib import Path

MANIFEST_PATH = Path("reports/eval/current_artifact_manifest.json")

REQUIRED_TOP_KEYS = {"generated_at", "corpus", "knowledge_graph", "qrels"}
REQUIRED_CORPUS_KEYS = {"path", "row_count", "unique_source_ids"}
REQUIRED_QRELS_KEYS = {"gold", "silver", "bronze", "field_state_adjudicated"}
REQUIRED_QREL_ENTRY_KEYS = {"path", "rows", "available"}


def _load() -> dict:
    assert MANIFEST_PATH.exists(), (
        f"{MANIFEST_PATH} not found. Run: python scripts/build_artifact_manifest.py "
        "or ensure the file is present in the repo."
    )
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def test_manifest_parses():
    data = _load()
    assert isinstance(data, dict)


def test_manifest_has_required_top_keys():
    data = _load()
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_corpus_section_has_required_keys():
    data = _load()
    corpus = data.get("corpus", {})
    missing = REQUIRED_CORPUS_KEYS - set(corpus.keys())
    assert not missing, f"Corpus section missing keys: {missing}"


def test_corpus_row_count_is_positive():
    data = _load()
    row_count = data["corpus"].get("row_count", 0)
    assert row_count > 0, f"corpus.row_count must be positive, got {row_count}"


def test_corpus_unique_ids_is_positive():
    data = _load()
    uid = data["corpus"].get("unique_source_ids", 0)
    assert uid > 0, f"corpus.unique_source_ids must be positive, got {uid}"


def test_knowledge_graph_section_present():
    data = _load()
    kg = data.get("knowledge_graph", {})
    assert isinstance(kg, dict), "knowledge_graph must be a dict"


def test_qrels_section_has_required_keys():
    data = _load()
    qrels = data.get("qrels", {})
    missing = REQUIRED_QRELS_KEYS - set(qrels.keys())
    assert not missing, f"qrels section missing keys: {missing}"


def test_each_qrel_entry_has_required_fields():
    data = _load()
    qrels = data.get("qrels", {})
    for name, entry in qrels.items():
        missing = REQUIRED_QREL_ENTRY_KEYS - set(entry.keys())
        assert not missing, f"qrels.{name} missing fields: {missing}"


def test_qrel_rows_are_non_negative():
    data = _load()
    qrels = data.get("qrels", {})
    for name, entry in qrels.items():
        rows = entry.get("rows", -1)
        assert rows >= 0, f"qrels.{name}.rows must be >= 0, got {rows}"


def test_generated_at_is_string():
    data = _load()
    assert isinstance(data.get("generated_at"), str), "generated_at must be a string"


def test_corpus_path_points_to_v09():
    """Ensure the canonical path explicitly references full_corpus_v09.jsonl, not the directory."""
    data = _load()
    path = data["corpus"].get("path", "")
    assert "full_corpus_v09" in path, (
        f"corpus.path should reference full_corpus_v09.jsonl, got: {path}"
    )
