"""Tests for ablation runner."""
import tempfile
from pathlib import Path
import pytest
from neural_search.evaluation.ablation_runner import (
    AblationVariant,
    AblationConfig,
    CandidatePool,
    run_ablation,
    AblationReport,
    VARIANT_NAMES,
)
from neural_search.evaluation.usefulness_benchmark import (
    UsefulnessQuery,
    PairLabel,
)
from neural_search.retrieval.usefulness_scorer import DatasetContext


def _make_pool():
    return CandidatePool(
        candidates={
            "c1": DatasetContext("c1", modalities=["neuropixels"], tasks=["decision_making"],
                                 affordances=["choice_decoding"], data_standards=["nwb"],
                                 quality_score=0.9, trial_count=5000),
            "c2": DatasetContext("c2", modalities=["calcium_imaging"], tasks=["decision_making"],
                                 affordances=["dimensionality_reduction"], data_standards=["bids"],
                                 quality_score=0.7, trial_count=2000),
            "c3": DatasetContext("c3", modalities=["neuropixels"], tasks=["go_nogo"],
                                 affordances=["choice_decoding", "q_learning"],
                                 data_standards=["nwb"], quality_score=0.8, trial_count=8000),
            "c4": DatasetContext("c4", modalities=["eeg"], tasks=["rest"],
                                 affordances=[], data_standards=[], quality_score=0.3),
        }
    )


def _make_queries_labels():
    queries = [
        UsefulnessQuery(
            query_id="q1",
            query="neuropixels decision making choice decoding",
            intent="strict_lookup",
            candidate_ids=["c1", "c2", "c3", "c4"],
        )
    ]
    labels = [
        PairLabel(query_id="q1", candidate_id="c1", usefulness_label="highly_useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c3", usefulness_label="useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c2", usefulness_label="weakly_useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c4", usefulness_label="not_useful", label_type="reusable",
                  is_hard_negative=True),
    ]
    return queries, labels


class TestVariantNames:
    def test_all_expected_variants_present(self):
        expected = {
            "bm25_only", "dense_only", "graph_only", "affordance_only",
            "bm25_dense_rrf", "hybrid_static", "hybrid_intent_aware", "latent_usefulness_v08",
        }
        assert set(VARIANT_NAMES) == expected
        assert len(VARIANT_NAMES) == 8


class TestRunAblation:
    def test_report_has_all_variants(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        assert isinstance(report, AblationReport)
        for name in VARIANT_NAMES:
            assert name in report.variant_metrics, f"Missing variant: {name}"

    def test_all_metrics_bounded(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        for variant, metrics in report.variant_metrics.items():
            for m, v in metrics.items():
                assert 0.0 <= v <= 1.0, f"{variant}.{m} = {v} out of [0,1]"

    def test_markdown_report_contains_table(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        md = report.to_markdown()
        assert "|" in md
        assert "NDCG" in md.upper()

    def test_markdown_report_written_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queries, labels = _make_queries_labels()
            pool = _make_pool()
            out_path = Path(tmpdir) / "ablation.md"
            config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4, out_path=out_path)
            run_ablation(config)
            assert out_path.exists()
            content = out_path.read_text()
            assert "|" in content

    def test_latent_usefulness_variant_present_in_output(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        assert "latent_usefulness_v08" in report.variant_metrics
        m = report.variant_metrics["latent_usefulness_v08"]
        assert "ndcg_at_k" in m
        assert "mrr" in m
