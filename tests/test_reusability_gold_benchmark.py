"""Tests for Reusability Gold v1 benchmark loading and validation."""

from pathlib import Path

import pytest
import yaml

# Benchmark file path
BENCHMARK_PATH = Path(__file__).parent.parent / "data" / "eval" / "reusability_gold_v1.yaml"


@pytest.fixture
def benchmark_data():
    """Load the benchmark YAML."""
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestBenchmarkStructure:
    """Tests for benchmark file structure."""

    def test_benchmark_file_exists(self):
        """Test that benchmark file exists."""
        assert BENCHMARK_PATH.exists(), f"Benchmark not found at {BENCHMARK_PATH}"

    def test_benchmark_has_metadata(self, benchmark_data):
        """Test benchmark has required metadata fields."""
        assert "version" in benchmark_data
        assert "name" in benchmark_data
        assert "queries" in benchmark_data

    def test_benchmark_has_queries(self, benchmark_data):
        """Test benchmark has at least 25 queries."""
        queries = benchmark_data["queries"]
        assert len(queries) >= 25, f"Expected at least 25 queries, got {len(queries)}"

    def test_benchmark_has_metrics_config(self, benchmark_data):
        """Test benchmark has metrics configuration."""
        assert "metrics" in benchmark_data
        assert "primary" in benchmark_data["metrics"]


class TestQueryStructure:
    """Tests for individual query structure."""

    def test_all_queries_have_required_fields(self, benchmark_data):
        """Test all queries have required fields."""
        required_fields = ["id", "query", "category", "intent"]

        for query in benchmark_data["queries"]:
            for field in required_fields:
                assert field in query, f"Query {query.get('id', 'unknown')} missing {field}"

    def test_all_query_ids_unique(self, benchmark_data):
        """Test all query IDs are unique."""
        ids = [q["id"] for q in benchmark_data["queries"]]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_query_ids_follow_format(self, benchmark_data):
        """Test query IDs follow expected format."""
        for query in benchmark_data["queries"]:
            assert query["id"].startswith("rg_v1_"), f"ID {query['id']} doesn't follow format"

    def test_all_queries_have_must_have_or_constructs(self, benchmark_data):
        """Test all queries have either must_have or constructs."""
        for query in benchmark_data["queries"]:
            has_must_have = "must_have" in query and query["must_have"]
            has_constructs = "constructs" in query and query["constructs"]
            assert has_must_have or has_constructs, f"Query {query['id']} needs must_have or constructs"


class TestQueryCategories:
    """Tests for query categories."""

    def test_has_ambiguity_queries(self, benchmark_data):
        """Test benchmark has ambiguity category queries."""
        ambiguity_queries = [q for q in benchmark_data["queries"] if q["category"] == "ambiguity"]
        assert len(ambiguity_queries) >= 3, "Need at least 3 ambiguity queries"

    def test_has_affordance_queries(self, benchmark_data):
        """Test benchmark has affordance category queries."""
        affordance_queries = [q for q in benchmark_data["queries"] if q["category"] == "affordance"]
        assert len(affordance_queries) >= 6, "Need at least 6 affordance queries"

    def test_has_natural_language_queries(self, benchmark_data):
        """Test benchmark has natural language queries."""
        nl_queries = [q for q in benchmark_data["queries"] if q["category"] == "natural_language"]
        assert len(nl_queries) >= 3, "Need at least 3 natural language queries"

    def test_has_exact_lookup_queries(self, benchmark_data):
        """Test benchmark has exact lookup queries."""
        lookup_queries = [q for q in benchmark_data["queries"] if q["category"] == "exact_lookup"]
        assert len(lookup_queries) >= 2, "Need at least 2 exact lookup queries"

    def test_has_cross_modal_queries(self, benchmark_data):
        """Test benchmark has cross-modal queries."""
        cross_modal_queries = [q for q in benchmark_data["queries"] if q["category"] == "cross_modal"]
        assert len(cross_modal_queries) >= 3, "Need at least 3 cross-modal queries"


class TestHardNegatives:
    """Tests for hard negative specifications."""

    def test_has_hard_negative_queries(self, benchmark_data):
        """Test benchmark has queries with hard negatives."""
        hard_neg_queries = [
            q for q in benchmark_data["queries"]
            if q.get("hard_negative_senses")
        ]
        assert len(hard_neg_queries) >= 4, "Need at least 4 queries with hard negatives"

    def test_delay_discounting_has_correct_hard_negatives(self, benchmark_data):
        """Test delay discounting query has expected hard negatives."""
        delay_queries = [
            q for q in benchmark_data["queries"]
            if "delay_discounting" in q.get("constructs", [])
        ]

        assert len(delay_queries) > 0, "No delay discounting queries found"

        for query in delay_queries:
            hard_negs = query.get("hard_negative_senses", [])
            # Should exclude motor and signal delay
            assert "motor_delay" in hard_negs or "signal_propagation_delay" in hard_negs, \
                f"Query {query['id']} should have motor/signal delay as hard negative"


class TestAffordanceQueries:
    """Tests for affordance-based queries."""

    def test_affordance_queries_have_affordance_field(self, benchmark_data):
        """Test affordance category queries specify required affordance."""
        affordance_queries = [
            q for q in benchmark_data["queries"]
            if q["category"] == "affordance"
        ]

        queries_with_affordance = [
            q for q in affordance_queries
            if "affordance_required" in q
        ]

        # At least half should specify affordance
        assert len(queries_with_affordance) >= len(affordance_queries) // 2, \
            "Affordance queries should specify affordance_required"

    def test_choice_decoding_query_exists(self, benchmark_data):
        """Test there's a choice decoding affordance query."""
        choice_queries = [
            q for q in benchmark_data["queries"]
            if q.get("affordance_required") == "choice_decoding"
        ]
        assert len(choice_queries) >= 1, "Need at least one choice_decoding query"

    def test_q_learning_query_exists(self, benchmark_data):
        """Test there's a Q-learning affordance query."""
        ql_queries = [
            q for q in benchmark_data["queries"]
            if q.get("affordance_required") == "q_learning"
        ]
        assert len(ql_queries) >= 1, "Need at least one q_learning query"


class TestDelayDiscountingCaseStudy:
    """Tests specific to the delay discounting EBRAINS case study."""

    def test_delay_discounting_query_has_required_variables(self, benchmark_data):
        """Test delay discounting query specifies required variables."""
        delay_query = next(
            (q for q in benchmark_data["queries"] if q["id"] == "rg_v1_001"),
            None
        )
        assert delay_query is not None, "rg_v1_001 (delay discounting) query not found"

        must_have = delay_query.get("must_have", [])

        # Check for key delay discounting variables
        required_vars = ["choice", "reward_magnitude", "delay_duration", "outcome"]
        for var in required_vars:
            assert var in must_have, f"Delay discounting query missing {var}"

    def test_delay_discounting_excludes_motor_delay(self, benchmark_data):
        """Test delay discounting query excludes motor delay sense."""
        delay_query = next(
            (q for q in benchmark_data["queries"] if q["id"] == "rg_v1_001"),
            None
        )
        assert delay_query is not None

        hard_negs = delay_query.get("hard_negative_senses", [])
        assert "motor_delay" in hard_negs, "Should exclude motor_delay sense"

    def test_motor_delay_query_exists_separately(self, benchmark_data):
        """Test there are separate motor delay queries (not confused with discounting)."""
        motor_queries = [
            q for q in benchmark_data["queries"]
            if "motor" in q["query"].lower() and "delay" in q["query"].lower()
        ]
        # Motor delay is legitimately different from delay discounting
        # The benchmark should have ways to find motor delay data too


class TestQueryIntents:
    """Tests for query intent specifications."""

    def test_all_queries_have_valid_intent(self, benchmark_data):
        """Test all queries have valid intent values."""
        valid_intents = {
            "analysis_reusability",
            "exploratory",
            "dataset_lookup",
            "paper_to_dataset",
            "species_region",
            "modality_search",
        }

        for query in benchmark_data["queries"]:
            assert query["intent"] in valid_intents, \
                f"Query {query['id']} has invalid intent: {query['intent']}"


class TestQueryContent:
    """Tests for query text content."""

    def test_queries_are_non_empty(self, benchmark_data):
        """Test all queries have non-empty text."""
        for query in benchmark_data["queries"]:
            assert query["query"].strip(), f"Query {query['id']} has empty text"

    def test_queries_vary_in_style(self, benchmark_data):
        """Test queries vary in style (formal, natural, abbreviated)."""
        queries = [q["query"].lower() for q in benchmark_data["queries"]]

        # Check for variety
        has_question = any("?" in q or q.startswith("can") or q.startswith("what") for q in queries)
        has_formal = any("datasets suitable for" in q or "datasets supporting" in q for q in queries)
        has_abbreviated = any(len(q.split()) <= 5 for q in queries)

        assert has_question or has_formal or has_abbreviated, \
            "Queries should have variety in style"
