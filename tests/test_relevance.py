"""Tests for human relevance labeling infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_search.evaluation.relevance import (
    HumanEvaluationMetrics,
    RelevanceJudgment,
    RelevanceLabelSet,
    compute_hard_negative_violations,
    compute_human_evaluation_metrics,
    compute_human_precision,
    compute_human_recall,
    compute_mrr,
    compute_ndcg,
    create_judgment,
    load_relevance_labels,
    save_relevance_labels,
)


class TestRelevanceJudgment:
    """Tests for RelevanceJudgment dataclass."""

    def test_create_valid_judgment(self):
        """Test creating a valid relevance judgment."""
        judgment = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="reversal learning neuropixels",
            dataset_id="000026",
            dataset_title="Steinmetz et al. 2019",
            relevance="exact",
            task_match=3,
            modality_match=3,
            species_match=3,
            analysis_fit=3,
            reviewer_id="test_user",
            review_timestamp="2024-01-15T10:30:00Z",
            review_notes="Perfect match",
            confidence=0.95,
        )

        assert judgment.judgment_id == "j_001"
        assert judgment.relevance == "exact"
        assert judgment.task_match == 3

    def test_invalid_score_raises_error(self):
        """Test that invalid dimension scores raise ValueError."""
        with pytest.raises(ValueError, match="task_match must be between 0 and 3"):
            RelevanceJudgment(
                judgment_id="j_001",
                query_id="q_001",
                query_text="test",
                dataset_id="ds001",
                dataset_title="Test",
                relevance="relevant",
                task_match=5,  # Invalid: > 3
            )

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            RelevanceJudgment(
                judgment_id="j_001",
                query_id="q_001",
                query_text="test",
                dataset_id="ds001",
                dataset_title="Test",
                relevance="relevant",
                confidence=1.5,  # Invalid: > 1
            )

    def test_invalid_relevance_raises_error(self):
        """Test that invalid relevance level raises ValueError."""
        with pytest.raises(ValueError, match="relevance must be one of"):
            RelevanceJudgment(
                judgment_id="j_001",
                query_id="q_001",
                query_text="test",
                dataset_id="ds001",
                dataset_title="Test",
                relevance="super_relevant",  # Invalid
            )

    def test_relevance_score_property(self):
        """Test relevance_score property returns correct numeric value."""
        judgment = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="test",
            dataset_id="ds001",
            dataset_title="Test",
            relevance="exact",
        )
        assert judgment.relevance_score == 3

        judgment2 = RelevanceJudgment(
            judgment_id="j_002",
            query_id="q_001",
            query_text="test",
            dataset_id="ds002",
            dataset_title="Test 2",
            relevance="hard_negative",
        )
        assert judgment2.relevance_score == -2

    def test_dimension_score_total(self):
        """Test dimension_score_total property."""
        judgment = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="test",
            dataset_id="ds001",
            dataset_title="Test",
            relevance="exact",
            task_match=3,
            modality_match=2,
            species_match=3,
            analysis_fit=2,
        )
        assert judgment.dimension_score_total == 10

    def test_is_relevant_exact(self):
        """Test is_relevant with exact minimum."""
        judgment = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="test",
            dataset_id="ds001",
            dataset_title="Test",
            relevance="exact",
        )
        assert judgment.is_relevant("exact") is True
        assert judgment.is_relevant("highly_relevant") is True
        assert judgment.is_relevant("relevant") is True

    def test_is_relevant_partial(self):
        """Test is_relevant with partial relevance."""
        judgment = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="test",
            dataset_id="ds001",
            dataset_title="Test",
            relevance="partially",
        )
        assert judgment.is_relevant("exact") is False
        assert judgment.is_relevant("partially") is True
        assert judgment.is_relevant("not_relevant") is True

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = RelevanceJudgment(
            judgment_id="j_001",
            query_id="q_001",
            query_text="reversal learning",
            dataset_id="000026",
            dataset_title="Steinmetz",
            relevance="highly_relevant",
            task_match=2,
            modality_match=3,
            species_match=2,
            analysis_fit=3,
            reviewer_id="expert",
            review_timestamp="2024-01-15T10:00:00Z",
            review_notes="Good match",
            confidence=0.8,
        )

        data = original.to_dict()
        restored = RelevanceJudgment.from_dict(data)

        assert restored.judgment_id == original.judgment_id
        assert restored.relevance == original.relevance
        assert restored.task_match == original.task_match
        assert restored.confidence == original.confidence


class TestRelevanceLabelSet:
    """Tests for RelevanceLabelSet."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create a sample label set with judgments."""
        label_set = RelevanceLabelSet(
            query_id="q_001",
            query_text="reversal learning neuropixels",
        )

        judgments = [
            RelevanceJudgment(
                judgment_id="j_001",
                query_id="q_001",
                query_text="reversal learning neuropixels",
                dataset_id="ds001",
                dataset_title="Dataset 1",
                relevance="exact",
            ),
            RelevanceJudgment(
                judgment_id="j_002",
                query_id="q_001",
                query_text="reversal learning neuropixels",
                dataset_id="ds002",
                dataset_title="Dataset 2",
                relevance="relevant",
            ),
            RelevanceJudgment(
                judgment_id="j_003",
                query_id="q_001",
                query_text="reversal learning neuropixels",
                dataset_id="ds003",
                dataset_title="Dataset 3",
                relevance="not_relevant",
            ),
            RelevanceJudgment(
                judgment_id="j_004",
                query_id="q_001",
                query_text="reversal learning neuropixels",
                dataset_id="ds004",
                dataset_title="Dataset 4",
                relevance="hard_negative",
            ),
        ]

        for j in judgments:
            label_set.add_judgment(j)

        return label_set

    def test_add_judgment(self, sample_label_set):
        """Test adding judgments to label set."""
        assert len(sample_label_set.judgments) == 4

    def test_add_judgment_wrong_query_id(self, sample_label_set):
        """Test that adding judgment with wrong query_id raises error."""
        wrong_judgment = RelevanceJudgment(
            judgment_id="j_wrong",
            query_id="q_002",  # Wrong query ID
            query_text="different query",
            dataset_id="ds005",
            dataset_title="Dataset 5",
            relevance="relevant",
        )

        with pytest.raises(ValueError, match="does not match"):
            sample_label_set.add_judgment(wrong_judgment)

    def test_get_judgment_for_dataset(self, sample_label_set):
        """Test getting judgment for specific dataset."""
        judgment = sample_label_set.get_judgment_for_dataset("ds002")
        assert judgment is not None
        assert judgment.relevance == "relevant"

        missing = sample_label_set.get_judgment_for_dataset("ds999")
        assert missing is None

    def test_relevant_dataset_ids(self, sample_label_set):
        """Test getting relevant dataset IDs."""
        relevant = sample_label_set.relevant_dataset_ids
        assert "ds001" in relevant  # exact
        assert "ds002" in relevant  # relevant
        assert "ds003" not in relevant  # not_relevant
        assert "ds004" not in relevant  # hard_negative

    def test_exact_match_ids(self, sample_label_set):
        """Test getting exact match IDs."""
        exact = sample_label_set.exact_match_ids
        assert "ds001" in exact
        assert len(exact) == 1

    def test_hard_negative_ids(self, sample_label_set):
        """Test getting hard negative IDs."""
        hard_neg = sample_label_set.hard_negative_ids
        assert "ds004" in hard_neg
        assert len(hard_neg) == 1


class TestCreateJudgment:
    """Tests for create_judgment helper."""

    def test_creates_with_auto_id_and_timestamp(self):
        """Test that create_judgment auto-generates ID and timestamp."""
        judgment = create_judgment(
            query_id="q_001",
            query_text="test query",
            dataset_id="ds001",
            dataset_title="Test Dataset",
            relevance="relevant",
            reviewer_id="user123",
        )

        assert judgment.judgment_id.startswith("j_")
        assert len(judgment.judgment_id) > 5
        assert judgment.review_timestamp != ""
        assert "T" in judgment.review_timestamp  # ISO format


class TestSaveLoadRelevanceLabels:
    """Tests for saving and loading relevance labels."""

    def test_save_and_load_round_trip(self):
        """Test saving and loading preserves data."""
        judgments = [
            create_judgment(
                query_id="q_001",
                query_text="reversal learning",
                dataset_id="ds001",
                dataset_title="Dataset 1",
                relevance="exact",
                reviewer_id="user1",
                task_match=3,
                modality_match=2,
            ),
            create_judgment(
                query_id="q_001",
                query_text="reversal learning",
                dataset_id="ds002",
                dataset_title="Dataset 2",
                relevance="relevant",
                reviewer_id="user1",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            count = save_relevance_labels(judgments, temp_path, append=False)
            assert count == 2

            # Load
            label_sets = load_relevance_labels(temp_path)
            assert "q_001" in label_sets
            assert len(label_sets["q_001"].judgments) == 2

            # Verify data integrity
            loaded = label_sets["q_001"].get_judgment_for_dataset("ds001")
            assert loaded is not None
            assert loaded.relevance == "exact"
            assert loaded.task_match == 3
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns empty dict."""
        result = load_relevance_labels("/nonexistent/path.jsonl")
        assert result == {}

    def test_append_mode(self):
        """Test appending to existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # First save
            j1 = create_judgment(
                query_id="q_001",
                query_text="query 1",
                dataset_id="ds001",
                dataset_title="D1",
                relevance="exact",
                reviewer_id="user1",
            )
            save_relevance_labels([j1], temp_path, append=False)

            # Append
            j2 = create_judgment(
                query_id="q_001",
                query_text="query 1",
                dataset_id="ds002",
                dataset_title="D2",
                relevance="relevant",
                reviewer_id="user1",
            )
            save_relevance_labels([j2], temp_path, append=True)

            # Load and verify
            label_sets = load_relevance_labels(temp_path)
            assert len(label_sets["q_001"].judgments) == 2
        finally:
            temp_path.unlink(missing_ok=True)


class TestComputeHumanPrecision:
    """Tests for compute_human_precision."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create label set for precision tests."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")

        # ds001: exact, ds002: relevant, ds003: partially, ds004: not_relevant
        for ds_id, rel in [("ds001", "exact"), ("ds002", "relevant"),
                           ("ds003", "partially"), ("ds004", "not_relevant")]:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_precision_at_5_all_relevant(self, sample_label_set):
        """Test precision when all top results are relevant."""
        result_ids = ["ds001", "ds002", "ds001", "ds001", "ds002"]
        precision = compute_human_precision(result_ids, sample_label_set, k=5)
        assert precision == 1.0

    def test_precision_at_5_half_relevant(self, sample_label_set):
        """Test precision when half are relevant."""
        result_ids = ["ds001", "ds003", "ds002", "ds004", "ds003"]
        precision = compute_human_precision(result_ids, sample_label_set, k=5)
        # ds001 (exact) + ds002 (relevant) = 2 relevant / 5 = 0.4
        assert precision == 0.4

    def test_precision_exact_minimum(self, sample_label_set):
        """Test precision with exact minimum threshold."""
        result_ids = ["ds001", "ds002", "ds003", "ds004", "ds001"]
        precision = compute_human_precision(
            result_ids, sample_label_set, k=5, min_relevance="exact"
        )
        # Only ds001 (exact) counts = 2 occurrences / 5 = 0.4
        assert precision == 0.4

    def test_precision_empty_results(self, sample_label_set):
        """Test precision with empty results."""
        precision = compute_human_precision([], sample_label_set, k=5)
        assert precision == 0.0


class TestComputeHumanRecall:
    """Tests for compute_human_recall."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create label set for recall tests."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")

        # 3 relevant datasets (exact + highly_relevant + relevant)
        for ds_id, rel in [("ds001", "exact"), ("ds002", "highly_relevant"),
                           ("ds003", "relevant"), ("ds004", "not_relevant")]:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_recall_all_found(self, sample_label_set):
        """Test recall when all relevant are found."""
        result_ids = ["ds001", "ds002", "ds003", "ds004", "ds005"]
        recall = compute_human_recall(result_ids, sample_label_set)
        # All 3 relevant found / 3 relevant total = 1.0
        assert recall == 1.0

    def test_recall_partial(self, sample_label_set):
        """Test recall when some relevant are found."""
        result_ids = ["ds001", "ds004", "ds005"]
        recall = compute_human_recall(result_ids, sample_label_set)
        # 1 relevant found / 3 relevant total = 0.333
        assert recall == pytest.approx(1/3, abs=0.01)

    def test_recall_with_k_limit(self, sample_label_set):
        """Test recall with k limit."""
        result_ids = ["ds004", "ds001", "ds002", "ds003"]
        recall = compute_human_recall(result_ids, sample_label_set, k=2)
        # Only look at first 2: ds004 (not_relevant), ds001 (exact)
        # 1 relevant found / 3 relevant total
        assert recall == pytest.approx(1/3, abs=0.01)


class TestComputeHardNegativeViolations:
    """Tests for compute_hard_negative_violations."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create label set with hard negatives."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")

        for ds_id, rel in [("ds001", "exact"), ("ds002", "hard_negative"),
                           ("ds003", "hard_negative")]:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_no_violations(self, sample_label_set):
        """Test when no hard negatives in results."""
        result_ids = ["ds001", "ds004", "ds005"]
        violations = compute_hard_negative_violations(result_ids, sample_label_set)
        assert len(violations) == 0

    def test_one_violation(self, sample_label_set):
        """Test detecting single violation."""
        result_ids = ["ds001", "ds002", "ds004"]
        violations = compute_hard_negative_violations(result_ids, sample_label_set)
        assert len(violations) == 1
        assert "ds002" in violations

    def test_multiple_violations(self, sample_label_set):
        """Test detecting multiple violations."""
        result_ids = ["ds002", "ds001", "ds003"]
        violations = compute_hard_negative_violations(result_ids, sample_label_set)
        assert len(violations) == 2
        assert "ds002" in violations
        assert "ds003" in violations


class TestComputeMRR:
    """Tests for compute_mrr (Mean Reciprocal Rank)."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create label set for MRR tests."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")

        for ds_id, rel in [("ds001", "exact"), ("ds002", "relevant"),
                           ("ds003", "not_relevant")]:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_mrr_first_position(self, sample_label_set):
        """Test MRR when relevant is first."""
        result_ids = ["ds001", "ds003", "ds004"]
        mrr = compute_mrr(result_ids, sample_label_set)
        assert mrr == 1.0

    def test_mrr_second_position(self, sample_label_set):
        """Test MRR when relevant is second."""
        result_ids = ["ds003", "ds001", "ds004"]
        mrr = compute_mrr(result_ids, sample_label_set)
        assert mrr == 0.5

    def test_mrr_third_position(self, sample_label_set):
        """Test MRR when relevant is third."""
        result_ids = ["ds003", "ds004", "ds002"]
        mrr = compute_mrr(result_ids, sample_label_set)
        assert mrr == pytest.approx(1/3, abs=0.01)

    def test_mrr_no_relevant(self, sample_label_set):
        """Test MRR when no relevant in results."""
        result_ids = ["ds003", "ds004", "ds005"]
        mrr = compute_mrr(result_ids, sample_label_set)
        assert mrr == 0.0


class TestComputeNDCG:
    """Tests for compute_ndcg."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create label set for NDCG tests."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")

        # exact (gain=5), highly_relevant (gain=4), relevant (gain=3)
        for ds_id, rel in [("ds001", "exact"), ("ds002", "highly_relevant"),
                           ("ds003", "relevant"), ("ds004", "not_relevant")]:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_ndcg_perfect_ranking(self, sample_label_set):
        """Test NDCG with perfect ranking (best first)."""
        # Ideal order: ds001 (exact), ds002 (highly_relevant), ds003 (relevant)
        result_ids = ["ds001", "ds002", "ds003", "ds004"]
        ndcg = compute_ndcg(result_ids, sample_label_set, k=4)
        assert ndcg == 1.0

    def test_ndcg_reversed_ranking(self, sample_label_set):
        """Test NDCG with reversed ranking (worst first)."""
        result_ids = ["ds004", "ds003", "ds002", "ds001"]
        ndcg = compute_ndcg(result_ids, sample_label_set, k=4)
        # Should be less than 1.0 since ordering is suboptimal
        assert 0 < ndcg < 1.0

    def test_ndcg_no_judgments(self):
        """Test NDCG with empty label set."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test")
        result_ids = ["ds001", "ds002"]
        ndcg = compute_ndcg(result_ids, label_set, k=5)
        assert ndcg == 0.0


class TestComputeHumanEvaluationMetrics:
    """Tests for compute_human_evaluation_metrics."""

    @pytest.fixture
    def sample_label_set(self) -> RelevanceLabelSet:
        """Create comprehensive label set."""
        label_set = RelevanceLabelSet(query_id="q_001", query_text="test query")

        judgments = [
            ("ds001", "exact"),
            ("ds002", "highly_relevant"),
            ("ds003", "relevant"),
            ("ds004", "partially"),
            ("ds005", "not_relevant"),
            ("ds006", "hard_negative"),
        ]

        for ds_id, rel in judgments:
            j = RelevanceJudgment(
                judgment_id=f"j_{ds_id}",
                query_id="q_001",
                query_text="test query",
                dataset_id=ds_id,
                dataset_title=f"Dataset {ds_id}",
                relevance=rel,
            )
            label_set.add_judgment(j)

        return label_set

    def test_computes_all_metrics(self, sample_label_set):
        """Test that all metrics are computed."""
        result_ids = ["ds001", "ds002", "ds003", "ds004", "ds005",
                      "ds006", "ds007", "ds008", "ds009", "ds010"]

        metrics = compute_human_evaluation_metrics(result_ids, sample_label_set)

        assert isinstance(metrics, HumanEvaluationMetrics)
        assert metrics.query_id == "q_001"
        assert metrics.query_text == "test query"
        assert 0 <= metrics.precision_at_5 <= 1
        assert 0 <= metrics.precision_at_10 <= 1
        assert 0 <= metrics.recall_at_10 <= 1
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.ndcg_at_10 <= 1
        assert metrics.hard_negative_violations >= 0
        assert metrics.total_judgments == 6

    def test_detects_hard_negative_violations(self, sample_label_set):
        """Test hard negative violation detection in metrics."""
        # ds006 is hard_negative, include it in top 10
        result_ids = ["ds006", "ds001", "ds002"]

        metrics = compute_human_evaluation_metrics(result_ids, sample_label_set)

        assert metrics.hard_negative_violations == 1
        assert "ds006" in metrics.hard_negative_ids

    def test_counts_relevant_judgments(self, sample_label_set):
        """Test counting relevant judgments."""
        result_ids = ["ds001"]
        metrics = compute_human_evaluation_metrics(result_ids, sample_label_set)

        # exact + highly_relevant + relevant = 3
        assert metrics.relevant_judgments == 3
        assert metrics.exact_match_judgments == 1
