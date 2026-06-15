"""Tests for source diversity reranking in Sprint 4."""
from __future__ import annotations

from neural_search.schemas import SearchResult
from neural_search.search.diversity import apply_source_diversity, diversity_stats


def _result(dataset_id: str, source: str, score: float) -> SearchResult:
    return SearchResult(dataset_id=dataset_id, source=source, score=score)


def _ranked(*pairs: tuple[str, str, float]) -> list[SearchResult]:
    """Build a pre-sorted results list from (dataset_id, source, score) tuples."""
    results = [_result(did, src, sc) for did, src, sc in pairs]
    results.sort(key=lambda r: r.score, reverse=True)
    return results


class TestApplySourceDiversity:
    def test_no_op_when_all_sources_distinct(self) -> None:
        results = _ranked(
            ("d1", "dandi", 0.9),
            ("d2", "openneuro", 0.8),
            ("d3", "neurovault", 0.7),
        )
        out = apply_source_diversity(results, max_per_source=3)
        assert [r.dataset_id for r in out] == ["d1", "d2", "d3"]

    def test_defers_excess_source_entries(self) -> None:
        results = _ranked(
            ("d1", "neurovault", 0.95),
            ("d2", "neurovault", 0.90),
            ("d3", "neurovault", 0.85),
            ("d4", "neurovault", 0.80),  # 4th from neurovault — should be deferred
            ("d5", "openneuro", 0.75),
        )
        out = apply_source_diversity(results, max_per_source=3)
        accepted_ids = [r.dataset_id for r in out[:4]]
        assert "d4" not in accepted_ids
        assert "d5" in accepted_ids
        # d4 is deferred to end
        assert out[-1].dataset_id == "d4"

    def test_limit_truncates_combined_list(self) -> None:
        results = _ranked(
            ("d1", "nv", 0.9),
            ("d2", "nv", 0.8),
            ("d3", "nv", 0.7),
            ("d4", "nv", 0.6),
            ("d5", "other", 0.5),
        )
        out = apply_source_diversity(results, max_per_source=2, limit=3)
        assert len(out) == 3
        # First 2 from nv, then 1 from other (d4 deferred but cutoff by limit)
        assert out[0].dataset_id == "d1"
        assert out[1].dataset_id == "d2"
        assert out[2].dataset_id == "d5"

    def test_max_per_source_zero_preserves_all(self) -> None:
        results = _ranked(
            ("d1", "s", 0.9),
            ("d2", "s", 0.8),
        )
        out = apply_source_diversity(results, max_per_source=0)
        assert len(out) == 2

    def test_max_per_source_one_strict_diversity(self) -> None:
        results = _ranked(
            ("d1", "nv", 0.9),
            ("d2", "dn", 0.8),
            ("d3", "nv", 0.7),
            ("d4", "dn", 0.6),
        )
        out = apply_source_diversity(results, max_per_source=1)
        assert out[0].dataset_id == "d1"
        assert out[1].dataset_id == "d2"
        # d3 and d4 deferred (second from their sources)
        assert {r.dataset_id for r in out[2:]} == {"d3", "d4"}

    def test_empty_input(self) -> None:
        assert apply_source_diversity([]) == []

    def test_single_source_not_exceeded(self) -> None:
        results = _ranked(
            ("d1", "dandi", 0.9),
            ("d2", "dandi", 0.8),
        )
        out = apply_source_diversity(results, max_per_source=3)
        assert len(out) == 2
        assert [r.dataset_id for r in out] == ["d1", "d2"]

    def test_unknown_source_treated_as_single_bucket(self) -> None:
        results = [
            _result("d1", "", 0.9),
            _result("d2", "", 0.8),
            _result("d3", "", 0.7),
            _result("d4", "", 0.6),
        ]
        out = apply_source_diversity(results, max_per_source=2, limit=4)
        assert out[0].dataset_id == "d1"
        assert out[1].dataset_id == "d2"
        assert out[2].dataset_id == "d3"  # deferred, appended after
        assert out[3].dataset_id == "d4"

    def test_score_order_preserved_within_accepted(self) -> None:
        results = _ranked(
            ("d1", "a", 0.95),
            ("d2", "b", 0.90),
            ("d3", "a", 0.85),
            ("d4", "c", 0.80),
            ("d5", "a", 0.75),
            ("d6", "d", 0.70),
        )
        out = apply_source_diversity(results, max_per_source=2)
        accepted = out[:5]  # d5 (third 'a') is deferred
        scores = [r.score for r in accepted]
        assert scores == sorted(scores, reverse=True)

    def test_limit_none_returns_all(self) -> None:
        results = _ranked(
            ("d1", "s", 0.9),
            ("d2", "s", 0.8),
            ("d3", "s", 0.7),
        )
        out = apply_source_diversity(results, max_per_source=2, limit=None)
        assert len(out) == 3


class TestDiversityStats:
    def test_counts_per_source(self) -> None:
        results = [
            _result("d1", "dandi", 0.9),
            _result("d2", "dandi", 0.8),
            _result("d3", "openneuro", 0.7),
        ]
        stats = diversity_stats(results)
        assert stats["dandi"] == 2
        assert stats["openneuro"] == 1

    def test_sorted_descending(self) -> None:
        results = [
            _result("d1", "a", 0.9),
            _result("d2", "b", 0.8),
            _result("d3", "b", 0.7),
            _result("d4", "b", 0.6),
        ]
        stats = diversity_stats(results)
        counts = list(stats.values())
        assert counts == sorted(counts, reverse=True)

    def test_empty_source_mapped_to_unknown(self) -> None:
        results = [_result("d1", "", 0.9)]
        stats = diversity_stats(results)
        assert "unknown" in stats
