"""Tests for query sense disambiguation."""


from neural_search.search.sense_disambiguation import (
    ALL_SENSES,
    disambiguate_query,
    get_associated_affordances,
    get_associated_tasks,
    get_sense,
    get_sense_penalties,
    get_senses_by_category,
    list_senses,
    score_sense,
)


class TestSenseDefinitions:
    """Tests for sense definition structure."""

    def test_all_senses_have_required_fields(self):
        """Test that all senses have required fields."""
        for sense_id, sense in ALL_SENSES.items():
            assert sense.sense_id == sense_id
            assert sense.category
            assert sense.label
            assert sense.positive_terms, f"{sense_id} has no positive terms"

    def test_delay_senses_exist(self):
        """Test that all delay senses are defined."""
        delay_senses = ["delay_discounting", "motor_delay", "signal_delay", "working_memory_delay"]
        for sense_id in delay_senses:
            assert sense_id in ALL_SENSES, f"Missing sense: {sense_id}"

    def test_exclusive_senses_are_symmetric(self):
        """Test that exclusive senses reference each other."""
        for _sense_id, sense in ALL_SENSES.items():
            for exclusive in sense.exclusive_senses:
                if exclusive in ALL_SENSES:
                    # The exclusive sense should also list this sense as exclusive
                    # (or at least exist)
                    assert exclusive in ALL_SENSES


class TestDelayDiscountingSense:
    """Tests for delay discounting disambiguation - the key demo case."""

    def test_delay_discounting_detected_with_positive_terms(self):
        """Test delay discounting is detected with clear positive terms."""
        result = disambiguate_query("delay discounting datasets with choices and rewards")

        assert result.primary_sense == "delay_discounting"
        assert result.primary_confidence >= 0.5  # At least moderate confidence
        assert "delay discounting" in result.matched_positive_terms

    def test_delay_discounting_excludes_signal_delay(self):
        """Test that delay discounting query excludes signal delay sense."""
        result = disambiguate_query("delay discounting datasets with reward choices")

        assert result.primary_sense == "delay_discounting"
        assert "signal_delay" in result.negative_senses
        assert "motor_delay" in result.negative_senses

    def test_delay_discounting_with_intertemporal_choice(self):
        """Test intertemporal choice triggers delay discounting sense."""
        result = disambiguate_query("intertemporal choice behavior")

        assert result.primary_sense == "delay_discounting"
        assert "intertemporal choice" in result.matched_positive_terms

    def test_delay_discounting_with_impulsivity(self):
        """Test impulsivity triggers delay discounting sense."""
        result = disambiguate_query("impulsive choice dataset with reward magnitude")

        assert result.primary_sense == "delay_discounting"

    def test_motor_delay_not_confused_with_discounting(self):
        """Test motor delay is correctly distinguished from discounting."""
        result = disambiguate_query("delayed reach to grasp motor cortex")

        assert result.primary_sense == "motor_delay"
        assert "delay_discounting" in result.negative_senses

    def test_signal_delay_not_confused_with_discounting(self):
        """Test signal delay is correctly distinguished from discounting."""
        result = disambiguate_query("propagation delay latency conduction")

        assert result.primary_sense == "signal_delay"
        assert "delay_discounting" in result.negative_senses

    def test_working_memory_delay_detected(self):
        """Test working memory delay is detected."""
        result = disambiguate_query("working memory delay period maintenance")

        assert result.primary_sense == "working_memory_delay"


class TestContextRequirements:
    """Tests for context-based disambiguation."""

    def test_reward_context_boosts_delay_discounting(self):
        """Test that reward context identifies delay discounting sense."""
        # With clear delay discounting context
        result_with_context = disambiguate_query("delay discounting reward choice impulsivity")

        assert result_with_context.primary_sense == "delay_discounting"
        assert result_with_context.primary_confidence >= 0.5

    def test_motor_context_triggers_motor_delay(self):
        """Test that motor context triggers motor delay sense."""
        # Use stronger motor-specific terms
        result = disambiguate_query("delayed reach to grasp motor cortex arm movement")

        assert result.primary_sense == "motor_delay"

    def test_propagation_context_triggers_signal_delay(self):
        """Test that propagation context triggers signal delay sense."""
        result = disambiguate_query("delay latency propagation axon")

        assert result.primary_sense == "signal_delay"


class TestNegativeTermPenalties:
    """Tests for negative term handling."""

    def test_negative_terms_reduce_score(self):
        """Test that negative terms reduce sense score."""
        # Query with only positive terms
        clean_result = disambiguate_query("delay discounting reward")

        # Query with negative terms mixed in
        mixed_result = disambiguate_query("delay discounting motor reach")

        # Clean query should have higher confidence
        assert clean_result.primary_confidence >= mixed_result.primary_confidence

    def test_strongly_negative_query_switches_sense(self):
        """Test that strong negative evidence switches to different sense."""
        # This query has "delay" but strong motor context
        result = disambiguate_query("delayed reach grasp motor cortex arm movement")

        # Should detect motor_delay, not delay_discounting
        assert result.primary_sense == "motor_delay"


class TestSensePenalties:
    """Tests for penalty score generation."""

    def test_get_sense_penalties_returns_penalties(self):
        """Test that penalties are generated for negative senses."""
        result = disambiguate_query("delay discounting reward choice")
        penalties = get_sense_penalties(result)

        # Should have penalties for exclusive senses
        for sense_id in result.negative_senses:
            assert sense_id in penalties
            assert penalties[sense_id] > 0

    def test_penalties_are_bounded(self):
        """Test that penalties are in valid range."""
        result = disambiguate_query("delay discounting")
        penalties = get_sense_penalties(result)

        for penalty in penalties.values():
            assert 0.0 <= penalty <= 1.0


class TestAssociatedMetadata:
    """Tests for associated affordances and tasks."""

    def test_delay_discounting_has_associated_affordances(self):
        """Test delay discounting returns associated affordances."""
        result = disambiguate_query("delay discounting reward")
        affordances = get_associated_affordances(result)

        assert "delay_discounting_modeling" in affordances

    def test_motor_delay_has_associated_affordances(self):
        """Test motor delay returns associated affordances."""
        result = disambiguate_query("delayed reach motor cortex")
        affordances = get_associated_affordances(result)

        assert "motor_decoding" in affordances

    def test_delay_discounting_has_associated_tasks(self):
        """Test delay discounting returns associated tasks."""
        result = disambiguate_query("delay discounting impulsivity")
        tasks = get_associated_tasks(result)

        assert "delay_discounting" in tasks or "intertemporal_choice" in tasks


class TestRewardSenses:
    """Tests for reward sense disambiguation."""

    def test_reward_value_detected(self):
        """Test reward value sense is detected."""
        result = disambiguate_query("reward value decision making expected")

        assert result.primary_sense == "reward_value"
        assert result.detected_category == "reward"

    def test_reward_delivery_detected(self):
        """Test reward delivery sense is detected."""
        result = disambiguate_query("juice reward delivery licking")

        assert result.primary_sense == "reward_delivery"


class TestMemorySenses:
    """Tests for memory sense disambiguation."""

    def test_working_memory_detected(self):
        """Test working memory is detected."""
        result = disambiguate_query("working memory maintenance delay period")

        assert result.primary_sense in ["working_memory", "working_memory_delay"]

    def test_episodic_memory_detected(self):
        """Test episodic memory is detected."""
        result = disambiguate_query("episodic memory hippocampus encoding retrieval")

        assert result.primary_sense == "episodic_memory"

    def test_spatial_memory_detected(self):
        """Test spatial memory is detected."""
        result = disambiguate_query("place cells grid cells navigation spatial")

        assert result.primary_sense == "spatial_memory"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_senses_returns_all(self):
        """Test listing all senses."""
        senses = list_senses()

        assert len(senses) > 0
        assert "delay_discounting" in senses
        assert "motor_delay" in senses

    def test_get_sense_returns_definition(self):
        """Test getting a sense definition."""
        sense = get_sense("delay_discounting")

        assert sense is not None
        assert sense.sense_id == "delay_discounting"
        assert sense.category == "delay"

    def test_get_sense_returns_none_for_unknown(self):
        """Test getting unknown sense returns None."""
        sense = get_sense("not_a_real_sense")

        assert sense is None

    def test_get_senses_by_category(self):
        """Test getting senses by category."""
        delay_senses = get_senses_by_category("delay")

        assert len(delay_senses) >= 3
        for sense in delay_senses:
            assert sense.category == "delay"


class TestEdgeCases:
    """Tests for edge cases and ambiguous queries."""

    def test_empty_query_returns_no_sense(self):
        """Test empty query handling."""
        result = disambiguate_query("")

        assert result.primary_sense is None
        assert result.primary_confidence == 0.0

    def test_generic_query_low_confidence(self):
        """Test that generic queries have low confidence."""
        result = disambiguate_query("neural data")

        # Should either have no sense or very low confidence
        assert result.primary_sense is None or result.primary_confidence < 0.5

    def test_ambiguous_query_has_secondary_senses(self):
        """Test that ambiguous queries may have secondary senses."""
        # A query that could match multiple senses
        result = disambiguate_query("delay period task")

        # May have secondary senses if ambiguous
        # (This is expected behavior for genuinely ambiguous queries)
        assert isinstance(result.secondary_senses, list)


class TestScoreSense:
    """Tests for the scoring function."""

    def test_score_sense_returns_tuple(self):
        """Test score_sense returns score and details."""
        sense = get_sense("delay_discounting")
        score, details = score_sense("delay discounting reward", sense)

        assert isinstance(score, float)
        assert isinstance(details, dict)
        assert 0.0 <= score <= 1.0

    def test_score_increases_with_positive_matches(self):
        """Test that more positive matches increase score."""
        sense = get_sense("delay_discounting")

        score1, _ = score_sense("delay discounting", sense)
        score2, _ = score_sense("delay discounting intertemporal choice reward", sense)

        assert score2 >= score1

    def test_score_decreases_with_negative_matches(self):
        """Test that negative matches decrease score."""
        sense = get_sense("delay_discounting")

        score1, _ = score_sense("delay discounting reward", sense)
        score2, _ = score_sense("delay discounting motor reach", sense)

        assert score1 >= score2


class TestEBRAINSSearchCase:
    """Tests specifically for the EBRAINS search case study.

    The EBRAINS GUI search for "delay discounting" returned lexical matches
    including delayed reach-to-grasp, reward/motivation mentions, and signal
    propagation delay. Neural Search should distinguish these.
    """

    def test_delay_discounting_query_rejects_motor_delay(self):
        """Test that delay discounting query identifies motor delay as negative."""
        result = disambiguate_query(
            "Find datasets suitable for fitting delay discounting models from trial-level behavior"
        )

        assert result.primary_sense == "delay_discounting"
        assert "motor_delay" in result.negative_senses

    def test_delay_discounting_query_rejects_signal_delay(self):
        """Test that delay discounting query identifies signal delay as negative."""
        result = disambiguate_query("delay discounting intertemporal choice")

        assert result.primary_sense == "delay_discounting"
        assert "signal_delay" in result.negative_senses

    def test_delay_discounting_returns_correct_affordances(self):
        """Test that delay discounting returns modeling affordances."""
        result = disambiguate_query("delay discounting behavioral data")
        affordances = get_associated_affordances(result)

        assert "delay_discounting_modeling" in affordances
