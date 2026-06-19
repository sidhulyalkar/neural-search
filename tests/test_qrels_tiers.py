"""Tests for qrels tier generation."""
from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class TestQrelsTierSeparation:
    """Integration test: the build script produces three separate tier files."""

    def test_tiers_are_non_overlapping_on_source_field(self, tmp_path):
        """Gold, silver, bronze must each have only their tier's source label."""
        gold = tmp_path / "qrels_gold.jsonl"
        silver = tmp_path / "qrels_silver.jsonl"
        bronze = tmp_path / "qrels_bronze.jsonl"

        import subprocess
        import sys
        pair_evidence = [
            {
                "query_id": "q1", "record_id": "dandi:1",
                "query": {
                    "query_id": "q1", "query_text": "human fmri",
                    "intent": "META_ANALYSIS", "scientific_goal": "x",
                    "required_modalities": ["fmri"], "preferred_modalities": [],
                    "required_species": ["human"], "preferred_species": [],
                    "brain_regions": [], "task_constraints": [],
                    "data_level_requirements": [], "hard_negatives": [],
                    "analysis_affordances": [],
                },
                "dataset": {
                    "record_id": "dandi:1", "source": "dandi", "title": "Human fMRI",
                    "description": "fMRI study", "species": ["human"],
                    "modalities": ["fmri"], "data_levels": ["raw"], "tasks": [],
                    "regions": [], "license": "CC-BY-4.0", "doi": None, "url": None,
                    "raw_data_available": True, "metadata_completeness": 0.9,
                    "has_behavior": True, "has_trials": True, "data_standards": ["NWB"],
                },
                "pooled_from": ["usefulness"], "min_rank": 1, "priority": "high",
            }
        ]
        _write_jsonl(tmp_path / "pair_evidence.jsonl", pair_evidence)
        _write_jsonl(tmp_path / "votes.jsonl", [])

        result = subprocess.run(
            [sys.executable, "scripts/eval/build_qrels_from_votes.py",
             "--evidence", str(tmp_path / "pair_evidence.jsonl"),
             "--votes", str(tmp_path / "votes.jsonl"),
             "--out-gold", str(gold),
             "--out-silver", str(silver),
             "--out-bronze", str(bronze),
             "--audit-queue", str(tmp_path / "audit_queue.jsonl"),
             ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        silver_rows = _read_jsonl(silver)
        bronze_rows = _read_jsonl(bronze)
        gold_rows = _read_jsonl(gold)

        assert gold_rows == []
        total = len(silver_rows) + len(bronze_rows)
        assert total == 1

        for r in silver_rows:
            assert r["source"] == "silver"
        for r in bronze_rows:
            assert r["source"] == "bronze"
