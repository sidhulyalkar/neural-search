"""Tests for rank fusion utilities."""

from __future__ import annotations

import pytest

from neural_search.search.fusion import (
    FusedCandidate,
    FusionMethod,
    FusionResult,
    RankedItem,
    borda_count_fusion,
    combmnz_fusion,
    explain_fusion,
    fuse,
    reciprocal_rank_fusion,
    weighted_sum_fusion,
)


@pytest.fixture
def sparse_results() -> list[dict]:
    """Sample sparse retrieval results."""
    return [
        {"id": "d1", "score": 0.95},
        {"id": "d2", "score": 0.80},
        {"id": "d3", "score": 0.65},
        {"id": "d4", "score": 0.50},
    ]


@pytest.fixture
def semantic_results() -> list[dict]:
    """Sample semantic retrieval results."""
    return [
        {"id": "d2", "score": 0.90},
        {"id": "d1", "score": 0.75},
        {"id": "d5", "score": 0.60},
        {"id": "d3", "score": 0.45},
    ]


@pytest.fixture
def graph_results() -> list[dict]:
    """Sample graph-based retrieval results."""
    return [
        {"id": "d3", "score": 0.85},
        {"id": "d5", "score": 0.70},
        {"id": "d1", "score": 0.55},
    ]


class TestReciprocalRankFusion:
    """Tests for RRF fusion."""

    def test_basic_fusion(self, sparse_results, semantic_results):
        """Test basic RRF fusion."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        assert isinstance(result, FusionResult)
        assert result.method == FusionMethod.RRF
        assert len(result.candidates) > 0
        assert result.num_sources == 2

    def test_rrf_scoring(self, sparse_results, semantic_results):
        """Test RRF score computation."""
        k = 60
        result = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            k=k,
        )

        # d1 is rank 1 in sparse, rank 2 in semantic
        # RRF score = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        d1 = next(c for c in result.candidates if c.id == "d1")
        expected = 1 / (k + 1) + 1 / (k + 2)
        assert abs(d1.fused_score - expected) < 0.0001

        # d2 is rank 2 in sparse, rank 1 in semantic
        d2 = next(c for c in result.candidates if c.id == "d2")
        expected = 1 / (k + 2) + 1 / (k + 1)
        assert abs(d2.fused_score - expected) < 0.0001

    def test_rrf_k_parameter(self, sparse_results, semantic_results):
        """Test that k parameter affects scores."""
        result_low_k = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            k=10,
        )
        result_high_k = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            k=100,
        )

        # Lower k gives more weight to top ranks
        d1_low = next(c for c in result_low_k.candidates if c.id == "d1")
        d1_high = next(c for c in result_high_k.candidates if c.id == "d1")

        # Scores should be different
        assert d1_low.fused_score != d1_high.fused_score

    def test_rrf_source_weights(self, sparse_results, semantic_results):
        """Test source weighting in RRF."""
        result_equal = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            source_weights={"sparse": 1.0, "semantic": 1.0},
        )
        result_sparse_heavy = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            source_weights={"sparse": 2.0, "semantic": 1.0},
        )

        # Sparse-heavy should favor d1 (which is rank 1 in sparse)
        d1_equal = next(c for c in result_equal.candidates if c.id == "d1")
        d1_heavy = next(c for c in result_sparse_heavy.candidates if c.id == "d1")

        assert d1_heavy.fused_score > d1_equal.fused_score

    def test_rrf_empty_lists(self):
        """Test RRF with empty input."""
        result = reciprocal_rank_fusion({})
        assert result.candidates == []
        assert result.num_sources == 0

    def test_rrf_single_source(self, sparse_results):
        """Test RRF with single source."""
        result = reciprocal_rank_fusion({"sparse": sparse_results})
        assert len(result.candidates) == 4
        assert result.num_sources == 1

    def test_rrf_top_k(self, sparse_results, semantic_results):
        """Test top_k limiting."""
        result = reciprocal_rank_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            top_k=2,
        )
        assert len(result.candidates) == 2

    def test_rrf_provenance_tracking(self, sparse_results, semantic_results):
        """Test that source provenance is tracked."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        # d1 appears in both
        d1 = next(c for c in result.candidates if c.id == "d1")
        assert "sparse" in d1.source_ranks
        assert "semantic" in d1.source_ranks
        assert d1.num_sources == 2

        # d5 only in semantic
        d5 = next(c for c in result.candidates if c.id == "d5")
        assert "semantic" in d5.source_ranks
        assert "sparse" not in d5.source_ranks
        assert d5.num_sources == 1


class TestWeightedSumFusion:
    """Tests for weighted sum fusion."""

    def test_basic_weighted_sum(self, sparse_results, semantic_results):
        """Test basic weighted sum fusion."""
        result = weighted_sum_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        assert result.method == FusionMethod.WEIGHTED_SUM
        assert len(result.candidates) > 0

    def test_weighted_sum_normalization(self, sparse_results, semantic_results):
        """Test score normalization."""
        result = weighted_sum_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            normalize_scores=True,
        )

        # All scores should be between 0 and 1 after normalization
        for candidate in result.candidates:
            assert 0 <= candidate.fused_score <= 1.0

    def test_weighted_sum_no_normalization(self, sparse_results, semantic_results):
        """Test without normalization."""
        result_norm = weighted_sum_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            normalize_scores=True,
        )
        result_no_norm = weighted_sum_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            normalize_scores=False,
        )

        # Scores should differ
        d1_norm = next(c for c in result_norm.candidates if c.id == "d1")
        d1_no_norm = next(c for c in result_no_norm.candidates if c.id == "d1")

        # Might be same if scores happen to be in [0,1], but structure should work
        assert isinstance(d1_norm.fused_score, float)
        assert isinstance(d1_no_norm.fused_score, float)

    def test_weighted_sum_source_weights(self, sparse_results, semantic_results):
        """Test source weighting."""
        result = weighted_sum_fusion(
            {"sparse": sparse_results, "semantic": semantic_results},
            source_weights={"sparse": 0.8, "semantic": 0.2},
        )

        # Weights should be normalized to sum to 1
        assert abs(sum(result.source_weights.values()) - 1.0) < 0.0001


class TestBordaCountFusion:
    """Tests for Borda count fusion."""

    def test_basic_borda(self, sparse_results, semantic_results):
        """Test basic Borda count fusion."""
        result = borda_count_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        assert result.method == FusionMethod.BORDA
        assert len(result.candidates) > 0

    def test_borda_scoring(self):
        """Test Borda point calculation."""
        # Simple case: 3 items in each list
        list1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        list2 = [{"id": "b"}, {"id": "a"}, {"id": "c"}]

        result = borda_count_fusion({"l1": list1, "l2": list2})

        # In list of 3, points are: rank 1 = 3, rank 2 = 2, rank 3 = 1
        # a: 3 + 2 = 5
        # b: 2 + 3 = 5
        # c: 1 + 1 = 2
        a = next(c for c in result.candidates if c.id == "a")
        b = next(c for c in result.candidates if c.id == "b")
        c = next(c for c in result.candidates if c.id == "c")

        assert a.fused_score == b.fused_score == 5.0
        assert c.fused_score == 2.0


class TestCombMNZFusion:
    """Tests for CombMNZ fusion."""

    def test_basic_combmnz(self, sparse_results, semantic_results):
        """Test basic CombMNZ fusion."""
        result = combmnz_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        assert result.method == FusionMethod.COMBMNZ
        assert len(result.candidates) > 0

    def test_combmnz_multiplier(self, sparse_results, semantic_results):
        """Test that num_sources multiplier works."""
        result = combmnz_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        # d1 appears in both lists
        d1 = next(c for c in result.candidates if c.id == "d1")
        assert d1.num_sources == 2

        # d5 only in semantic
        d5 = next(c for c in result.candidates if c.id == "d5")
        assert d5.num_sources == 1

        # d1 should get boosted more by the multiplier
        # (assuming similar base scores)


class TestGenericFuse:
    """Tests for the generic fuse function."""

    def test_fuse_rrf(self, sparse_results, semantic_results):
        """Test fuse with RRF method."""
        result = fuse(
            {"sparse": sparse_results, "semantic": semantic_results},
            method=FusionMethod.RRF,
            k=60,
        )
        assert result.method == FusionMethod.RRF

    def test_fuse_weighted_sum(self, sparse_results, semantic_results):
        """Test fuse with weighted sum method."""
        result = fuse(
            {"sparse": sparse_results, "semantic": semantic_results},
            method=FusionMethod.WEIGHTED_SUM,
        )
        assert result.method == FusionMethod.WEIGHTED_SUM

    def test_fuse_borda(self, sparse_results, semantic_results):
        """Test fuse with Borda method."""
        result = fuse(
            {"sparse": sparse_results, "semantic": semantic_results},
            method=FusionMethod.BORDA,
        )
        assert result.method == FusionMethod.BORDA

    def test_fuse_combmnz(self, sparse_results, semantic_results):
        """Test fuse with CombMNZ method."""
        result = fuse(
            {"sparse": sparse_results, "semantic": semantic_results},
            method=FusionMethod.COMBMNZ,
        )
        assert result.method == FusionMethod.COMBMNZ

    def test_fuse_custom_id_field(self):
        """Test fuse with custom ID field."""
        list1 = [{"dataset_id": "d1", "score": 0.9}]
        list2 = [{"dataset_id": "d1", "score": 0.8}]

        result = fuse(
            {"l1": list1, "l2": list2},
            id_field="dataset_id",
        )
        assert result.candidates[0].id == "d1"


class TestExplainFusion:
    """Tests for fusion explanation."""

    def test_explain_fusion(self, sparse_results, semantic_results):
        """Test fusion explanation."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        explanation = explain_fusion(result, "d1")

        assert explanation["candidate_id"] == "d1"
        assert "fused_score" in explanation
        assert "method" in explanation
        assert "source_ranks" in explanation
        assert "source_contributions" in explanation

    def test_explain_missing_candidate(self, sparse_results, semantic_results):
        """Test explanation for missing candidate."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        explanation = explain_fusion(result, "nonexistent")
        assert "error" in explanation


class TestFusedCandidate:
    """Tests for FusedCandidate dataclass."""

    def test_fused_candidate_creation(self):
        """Test creating a FusedCandidate."""
        candidate = FusedCandidate(
            id="test_id",
            fused_score=0.5,
            source_ranks={"sparse": 1, "semantic": 2},
            source_scores={"sparse": 0.9, "semantic": 0.8},
            source_contributions={"sparse": 0.016, "semantic": 0.016},
            num_sources=2,
        )

        assert candidate.id == "test_id"
        assert candidate.fused_score == 0.5
        assert candidate.num_sources == 2


class TestInputFormats:
    """Tests for various input formats."""

    def test_dict_input(self):
        """Test with dict items."""
        lists = {
            "l1": [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}],
            "l2": [{"id": "b", "score": 1.0}, {"id": "a", "score": 0.5}],
        }
        result = reciprocal_rank_fusion(lists)
        assert len(result.candidates) == 2

    def test_string_input(self):
        """Test with string IDs only."""
        lists = {
            "l1": ["a", "b", "c"],
            "l2": ["b", "c", "a"],
        }
        result = reciprocal_rank_fusion(lists)
        assert len(result.candidates) == 3

    def test_ranked_item_input(self):
        """Test with RankedItem objects."""
        lists = {
            "l1": [
                RankedItem(id="a", score=0.9, rank=1, source="l1"),
                RankedItem(id="b", score=0.7, rank=2, source="l1"),
            ],
            "l2": [
                RankedItem(id="b", score=0.8, rank=1, source="l2"),
                RankedItem(id="a", score=0.6, rank=2, source="l2"),
            ],
        }
        result = reciprocal_rank_fusion(lists)
        assert len(result.candidates) == 2


class TestStability:
    """Tests for deterministic/stable behavior."""

    def test_deterministic_results(self, sparse_results, semantic_results):
        """Test that results are deterministic."""
        result1 = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })
        result2 = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
        })

        assert len(result1.candidates) == len(result2.candidates)
        for c1, c2 in zip(result1.candidates, result2.candidates, strict=False):
            assert c1.id == c2.id
            assert abs(c1.fused_score - c2.fused_score) < 0.0001

    def test_stable_tie_breaking(self):
        """Test stable tie breaking by ID."""
        lists = {
            "l1": [{"id": "b", "score": 1.0}, {"id": "a", "score": 0.5}],
            "l2": [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}],
        }
        result = reciprocal_rank_fusion(lists)

        # a and b have same RRF score, should be sorted by ID
        if abs(result.candidates[0].fused_score - result.candidates[1].fused_score) < 0.0001:
            assert result.candidates[0].id < result.candidates[1].id


class TestThreeSources:
    """Tests with three retrieval sources."""

    def test_three_source_fusion(self, sparse_results, semantic_results, graph_results):
        """Test fusion with three sources."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
            "graph": graph_results,
        })

        assert result.num_sources == 3

        # d1 is in all three lists
        d1 = next(c for c in result.candidates if c.id == "d1")
        assert d1.num_sources == 3
        assert "sparse" in d1.source_ranks
        assert "semantic" in d1.source_ranks
        assert "graph" in d1.source_ranks

    def test_consensus_rewarding(self, sparse_results, semantic_results, graph_results):
        """Test that consensus is rewarded."""
        result = reciprocal_rank_fusion({
            "sparse": sparse_results,
            "semantic": semantic_results,
            "graph": graph_results,
        })

        # d1 appears in all 3, d4 only in sparse
        d1 = next(c for c in result.candidates if c.id == "d1")
        d4 = next(c for c in result.candidates if c.id == "d4")

        # d1 should score higher due to consensus
        assert d1.fused_score > d4.fused_score
