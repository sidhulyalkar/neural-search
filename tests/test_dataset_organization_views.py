"""Tests for build_dataset_organization_views.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _judgment(
    dataset_id: str,
    label: int,
    confidence: float = 0.8,
    ec: float = 0.7,
    hn: bool = False,
    abstain: bool = False,
    present: list[str] | None = None,
    missing: list[str] | None = None,
    failure_modes: list[str] | None = None,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "query_id": "q001",
        "label": label,
        "confidence": confidence,
        "evidence_completeness": ec,
        "hard_negative_detected": hn,
        "abstain_recommended": abstain,
        "required_dimensions_present": present or [],
        "required_dimensions_missing": missing or [],
        "failure_modes": failure_modes or [],
    }


def _ep(dataset_id: str, modalities: list[str] | None = None, species: list[str] | None = None) -> dict:
    return {
        "dataset_id": dataset_id,
        "title": f"Title for {dataset_id}",
        "has_raw_data": True,
        "has_processed_data": True,
        "dataset_modalities": modalities or ["extracellular_ephys"],
        "dataset_species": species or ["mouse"],
        "source_archive": "dandi",
    }


def _load_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_dataset_organization_views",
        Path(__file__).resolve().parents[1]
        / "scripts/eval/build_dataset_organization_views.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPerDatasetBest:
    def setup_method(self):
        self.mod = _load_script()

    def test_picks_highest_label(self):
        judgments = [
            _judgment("ds1", label=1, confidence=0.9),
            _judgment("ds1", label=3, confidence=0.7),
        ]
        best = self.mod._per_dataset_best(judgments)
        assert best["ds1"]["label"] == 3

    def test_breaks_ties_by_confidence(self):
        judgments = [
            _judgment("ds1", label=2, confidence=0.6),
            _judgment("ds1", label=2, confidence=0.9),
        ]
        best = self.mod._per_dataset_best(judgments)
        assert best["ds1"]["confidence"] == 0.9

    def test_missing_dataset_id_skipped(self):
        judgments = [{"label": 2, "confidence": 0.8}]
        best = self.mod._per_dataset_best(judgments)
        assert len(best) == 0


class TestModalityBucket:
    def setup_method(self):
        self.mod = _load_script()

    def test_ephys(self):
        assert self.mod._modality_bucket(["extracellular_ephys"]) == "electrophysiology"

    def test_calcium(self):
        assert self.mod._modality_bucket(["calcium_imaging"]) == "calcium_imaging"

    def test_fmri(self):
        assert self.mod._modality_bucket(["fmri"]) == "fmri_mri"

    def test_unknown(self):
        assert self.mod._modality_bucket([]) == "other"


class TestBuildViews:
    def setup_method(self):
        self.mod = _load_script()

    def _run(
        self,
        judgments: list[dict],
        evidence_packets: list[dict] | None = None,
    ) -> dict:
        return self.mod.build_views(judgments, {}, evidence_packets or [])

    def test_highly_relevant_contains_label_3(self):
        judgments = [
            _judgment("ds1", label=3),
            _judgment("ds2", label=1),
        ]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["relevance_tiers"]["highly_relevant"]]
        assert "ds1" in ids
        assert "ds2" not in ids

    def test_not_relevant_contains_label_0(self):
        judgments = [_judgment("ds1", label=0)]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["relevance_tiers"]["not_relevant"]]
        assert "ds1" in ids

    def test_hard_negative_appears_in_special_view(self):
        judgments = [_judgment("ds_hn", label=0, hn=True)]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["special_views"]["likely_hard_negatives"]]
        assert "ds_hn" in ids

    def test_abstain_high_label_in_needs_audit(self):
        judgments = [_judgment("ds_audit", label=2, abstain=True)]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["special_views"]["needs_human_audit"]]
        assert "ds_audit" in ids

    def test_raw_data_suitable_view(self):
        judgments = [_judgment("ds_raw", label=2)]
        eps = [_ep("ds_raw")]
        eps[0]["has_raw_data"] = True
        views = self._run(judgments, eps)
        ids = [e["dataset_id"] for e in views["special_views"]["raw_data_suitable"]]
        assert "ds_raw" in ids

    def test_high_ec_view(self):
        judgments = [_judgment("ds_hec", label=3, ec=0.9)]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["special_views"]["high_evidence_completeness"]]
        assert "ds_hec" in ids

    def test_low_ec_view(self):
        judgments = [_judgment("ds_lec", label=1, ec=0.2)]
        views = self._run(judgments)
        ids = [e["dataset_id"] for e in views["special_views"]["low_evidence_completeness"]]
        assert "ds_lec" in ids

    def test_modality_bucket_populated(self):
        judgments = [_judgment("ds_eph", label=2)]
        eps = [_ep("ds_eph", modalities=["extracellular_ephys"])]
        views = self._run(judgments, eps)
        assert "electrophysiology" in views["modality_buckets"]
        ids = [e["dataset_id"] for e in views["modality_buckets"]["electrophysiology"]]
        assert "ds_eph" in ids

    def test_disclaimer_present(self):
        views = self._run([_judgment("ds1", label=2)])
        assert "disclaimer" in views
        assert "neuro-judge" in views["disclaimer"].lower() or "llm" in views["disclaimer"].lower()

    def test_total_count_correct(self):
        judgments = [_judgment(f"ds{i}", label=i % 4) for i in range(7)]
        views = self._run(judgments)
        assert views["total_datasets_judged"] == 7


class TestRenderMarkdown:
    def setup_method(self):
        self.mod = _load_script()

    def test_markdown_contains_relevance_tiers(self):
        judgments = [
            _judgment("ds1", label=3),
            _judgment("ds2", label=1),
        ]
        views = self.mod.build_views(judgments, {}, [])
        md = self.mod.render_markdown(views)
        assert "Highly Relevant" in md
        assert "Weakly Related" in md

    def test_markdown_contains_disclaimer(self):
        views = self.mod.build_views([_judgment("ds1", label=2)], {}, [])
        md = self.mod.render_markdown(views)
        assert "IMPORTANT" in md or "disclaimer" in md.lower()


class TestMainCLI:
    def setup_method(self):
        self.mod = _load_script()

    def test_main_produces_json_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            judgments_path = tmpdir / "judgments.jsonl"
            with judgments_path.open("w") as fh:
                for j in [_judgment("ds1", label=3), _judgment("ds2", label=0)]:
                    fh.write(json.dumps(j) + "\n")

            json_out = tmpdir / "views.json"
            md_out = tmpdir / "views.md"
            self.mod.main([
                "--judgments", str(judgments_path),
                "--corpus-manifest", str(tmpdir / "nonexistent.json"),
                "--evidence-packets", str(tmpdir / "nonexistent.jsonl"),
                "--out-json", str(json_out),
                "--out-md", str(md_out),
            ])
            assert json_out.exists()
            assert md_out.exists()
            data = json.loads(json_out.read_text())
            assert data["total_datasets_judged"] == 2
