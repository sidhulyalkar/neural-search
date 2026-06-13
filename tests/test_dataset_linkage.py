"""Tests for dataset linkage benchmark."""

from __future__ import annotations

from neural_search.evaluation.dataset_linkage import (
    AnnotatorLabel,
    DatasetPair,
    LinkageBenchmark,
    LinkageMetrics,
    LinkageType,
    create_sample_benchmark,
    evaluate_linkage,
)


class TestDatasetPair:
    """Tests for DatasetPair model."""

    def test_basic_creation(self):
        """Test basic pair creation."""
        pair = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            linkage_type=LinkageType.SAME_TASK,
            relatedness_score=3,
        )

        assert pair.source_id == "dandi:000003"
        assert pair.target_id == "dandi:000005"
        assert pair.relatedness_score == 3

    def test_pair_id_generation(self):
        """Test automatic pair ID generation."""
        pair1 = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
        )
        pair2 = DatasetPair(
            source_id="dandi:000005",
            target_id="dandi:000003",
        )

        # Same pair in different order should get same ID
        assert pair1.pair_id == pair2.pair_id

    def test_annotator_labels(self):
        """Test adding annotator labels."""
        pair = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            annotator_labels=[
                AnnotatorLabel(
                    annotator_id="ann1",
                    linkage_type=LinkageType.SAME_TASK,
                    relatedness_score=3,
                ),
                AnnotatorLabel(
                    annotator_id="ann2",
                    linkage_type=LinkageType.SAME_TASK,
                    relatedness_score=2,
                ),
            ],
        )

        assert pair.n_annotators == 2

    def test_compute_agreement(self):
        """Test agreement computation."""
        pair = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            annotator_labels=[
                AnnotatorLabel(annotator_id="ann1", linkage_type=LinkageType.SAME_TASK, relatedness_score=3),
                AnnotatorLabel(annotator_id="ann2", linkage_type=LinkageType.SAME_TASK, relatedness_score=3),
            ],
        )

        agreement = pair.compute_agreement()
        assert agreement == 1.0  # Perfect agreement

    def test_compute_agreement_disagreement(self):
        """Test agreement with disagreement."""
        pair = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            annotator_labels=[
                AnnotatorLabel(annotator_id="ann1", linkage_type=LinkageType.SAME_TASK, relatedness_score=3),
                AnnotatorLabel(annotator_id="ann2", linkage_type=LinkageType.SAME_TASK, relatedness_score=1),
            ],
        )

        agreement = pair.compute_agreement()
        assert agreement == 0.0  # No agreement

    def test_compute_agreement_single_annotator(self):
        """Test agreement with single annotator returns None."""
        pair = DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            annotator_labels=[
                AnnotatorLabel(annotator_id="ann1", linkage_type=LinkageType.SAME_TASK, relatedness_score=3),
            ],
        )

        assert pair.compute_agreement() is None


class TestLinkageBenchmark:
    """Tests for LinkageBenchmark model."""

    def test_basic_creation(self):
        """Test basic benchmark creation."""
        benchmark = LinkageBenchmark(
            benchmark_id="test_v1",
            pairs=[
                DatasetPair(
                    source_id="dandi:000003",
                    target_id="dandi:000005",
                    relatedness_score=3,
                ),
            ],
        )

        assert benchmark.n_pairs == 1

    def test_get_pairs_by_type(self):
        """Test filtering pairs by type."""
        benchmark = LinkageBenchmark(
            benchmark_id="test_v1",
            pairs=[
                DatasetPair(source_id="d1", target_id="d2", linkage_type=LinkageType.SAME_TASK),
                DatasetPair(source_id="d3", target_id="d4", linkage_type=LinkageType.SAME_MODALITY),
                DatasetPair(source_id="d5", target_id="d6", linkage_type=LinkageType.SAME_TASK),
            ],
        )

        same_task = benchmark.get_pairs_by_type(LinkageType.SAME_TASK)
        assert len(same_task) == 2

    def test_get_source_ids(self):
        """Test getting unique source IDs."""
        benchmark = LinkageBenchmark(
            benchmark_id="test_v1",
            pairs=[
                DatasetPair(source_id="d1", target_id="d2"),
                DatasetPair(source_id="d1", target_id="d3"),
                DatasetPair(source_id="d2", target_id="d4"),
            ],
        )

        sources = benchmark.get_source_ids()
        assert sources == {"d1", "d2"}

    def test_to_relevance_labels(self):
        """Test conversion to relevance labels."""
        benchmark = LinkageBenchmark(
            benchmark_id="test_v1",
            pairs=[
                DatasetPair(source_id="d1", target_id="d2", relatedness_score=3),
                DatasetPair(source_id="d1", target_id="d3", relatedness_score=2),
                DatasetPair(source_id="d1", target_id="d4", relatedness_score=1),  # Below threshold
            ],
        )

        labels = benchmark.to_relevance_labels()

        assert "d1" in labels
        assert "d2" in labels["d1"]
        assert "d3" in labels["d1"]
        assert "d4" not in labels["d1"]  # Score < 2


class TestLinkageType:
    """Tests for LinkageType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert LinkageType.SAME_TASK == "same_task"
        assert LinkageType.SAME_MODALITY == "same_modality"
        assert LinkageType.UNRELATED == "unrelated"


class TestEvaluateLinkage:
    """Tests for linkage evaluation."""

    def test_perfect_retrieval(self):
        """Test evaluation with perfect retrieval."""
        benchmark = LinkageBenchmark(
            benchmark_id="test_v1",
            pairs=[
                DatasetPair(source_id="d1", target_id="d2", relatedness_score=3),
                DatasetPair(source_id="d1", target_id="d3", relatedness_score=2),
            ],
        )

        def perfect_retrieval(source_id: str) -> list[str]:
            return ["d2", "d3", "d4", "d5", "d6"]

        metrics = evaluate_linkage(perfect_retrieval, benchmark)

        assert metrics.precision_at_5 > 0
        assert metrics.mrr > 0

    def test_empty_benchmark(self):
        """Test evaluation with empty benchmark."""
        benchmark = LinkageBenchmark(benchmark_id="empty")

        def dummy_retrieval(source_id: str) -> list[str]:
            return []

        metrics = evaluate_linkage(dummy_retrieval, benchmark)

        assert metrics.n_queries == 0


class TestSampleBenchmark:
    """Tests for sample benchmark creation."""

    def test_create_sample(self):
        """Test sample benchmark creation."""
        benchmark = create_sample_benchmark()

        assert benchmark.n_pairs == 3
        assert benchmark.benchmark_id == "linkage_v1_sample"


class TestLinkageMetrics:
    """Tests for LinkageMetrics dataclass."""

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = LinkageMetrics(
            precision_at_5=0.8,
            mrr=0.9,
            n_queries=100,
        )

        d = metrics.to_dict()

        assert d["precision_at_5"] == 0.8
        assert d["mrr"] == 0.9
        assert d["n_queries"] == 100


class TestAnnotatorLabel:
    """Tests for AnnotatorLabel model."""

    def test_creation(self):
        """Test label creation."""
        label = AnnotatorLabel(
            annotator_id="ann1",
            linkage_type=LinkageType.SAME_TASK,
            relatedness_score=3,
            confidence=0.9,
            notes="Clear relationship",
        )

        assert label.annotator_id == "ann1"
        assert label.relatedness_score == 3
        assert label.confidence == 0.9
