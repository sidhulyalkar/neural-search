"""Tests for Boolean constraint parser."""

from __future__ import annotations

from neural_search.retrieval.constraint_parser import (
    ConstraintParser,
    ConstraintType,
    OperatorType,
    ParsedTerm,
    apply_negative_filter,
    parse_query,
)


class TestConstraintParser:
    """Tests for ConstraintParser."""

    def test_simple_query(self):
        """Test parsing a simple query without operators."""
        result = parse_query("mouse decision-making")

        assert "mouse" in result.positive_terms
        assert "decision-making" in result.positive_terms
        assert len(result.negative_terms) == 0

    def test_not_operator(self):
        """Test parsing NOT operator."""
        result = parse_query("ephys NOT fMRI")

        assert "ephys" in result.positive_terms
        assert "fMRI" in result.negative_terms

    def test_multiple_not(self):
        """Test multiple NOT operators."""
        result = parse_query("neural data NOT fMRI NOT EEG")

        assert "neural" in result.positive_terms
        assert "data" in result.positive_terms
        assert "fMRI" in result.negative_terms
        assert "EEG" in result.negative_terms

    def test_quoted_phrases(self):
        """Test quoted phrase handling."""
        result = parse_query('"visual cortex" NOT "motor cortex"')

        assert "visual cortex" in result.positive_terms
        assert "motor cortex" in result.negative_terms

    def test_and_operator(self):
        """Test AND operator (terms treated as positive)."""
        result = parse_query("mouse AND decision-making")

        assert "mouse" in result.positive_terms
        assert "decision-making" in result.positive_terms

    def test_case_insensitive_operators(self):
        """Test that operators are case-insensitive."""
        result1 = parse_query("ephys NOT fMRI")
        result2 = parse_query("ephys not fMRI")
        result3 = parse_query("ephys Not fMRI")

        assert result1.negative_terms == result2.negative_terms == result3.negative_terms

    def test_semantic_query_generation(self):
        """Test semantic query is built from positive terms."""
        result = parse_query("mouse decision-making NOT fMRI")

        assert "mouse" in result.semantic_query
        assert "decision-making" in result.semantic_query
        assert "fMRI" not in result.semantic_query

    def test_has_negation(self):
        """Test has_negation helper."""
        result1 = parse_query("mouse decision-making")
        result2 = parse_query("mouse NOT fMRI")

        assert not result1.has_negation()
        assert result2.has_negation()

    def test_original_query_preserved(self):
        """Test original query is preserved."""
        query = "complex query with operators NOT excluded"
        result = parse_query(query)

        assert result.original_query == query


class TestImplicitConstraints:
    """Tests for implicit constraint application."""

    def test_neuropixels_implies_ephys(self):
        """Test that 'neuropixels' implies ephys modality."""
        parser = ConstraintParser(apply_implicit_constraints=True)
        result = parser.parse("neuropixels recordings")

        assert len(result.implicit_constraints) > 0
        implicit = result.implicit_constraints[0]
        assert implicit.term == "ephys"
        assert implicit.source == "implicit"

    def test_two_photon_implies_calcium(self):
        """Test that 'two-photon' implies calcium imaging."""
        result = parse_query("two-photon imaging data")

        assert any(c.term == "calcium_imaging" for c in result.implicit_constraints)

    def test_disable_implicit_constraints(self):
        """Test disabling implicit constraints."""
        parser = ConstraintParser(apply_implicit_constraints=False)
        result = parser.parse("neuropixels recordings")

        assert len(result.implicit_constraints) == 0


class TestConstraintTree:
    """Tests for constraint tree building."""

    def test_parse_with_tree_or(self):
        """Test tree building with OR."""
        parser = ConstraintParser()
        result = parser.parse_with_tree("mouse OR rat")

        assert result.constraint_tree is not None
        assert result.constraint_tree.operator == OperatorType.OR

    def test_parse_with_tree_and(self):
        """Test tree building with AND."""
        parser = ConstraintParser()
        result = parser.parse_with_tree("ephys AND behavior")

        assert result.constraint_tree is not None
        assert result.constraint_tree.operator == OperatorType.AND

    def test_parse_with_tree_not(self):
        """Test tree building with NOT."""
        parser = ConstraintParser()
        result = parser.parse_with_tree("NOT fMRI")

        assert result.constraint_tree is not None
        assert result.constraint_tree.operator == OperatorType.NOT


class TestNegativeFilter:
    """Tests for negative term filtering."""

    def test_filter_excludes_matching(self):
        """Test that filter excludes matching results."""
        results = [
            {"title": "Visual cortex ephys", "description": "Neural data"},
            {"title": "fMRI study", "description": "BOLD imaging"},
            {"title": "Mouse behavior", "description": "Decision task"},
        ]

        filtered = apply_negative_filter(results, ["fMRI"])

        assert len(filtered) == 2
        assert all("fMRI" not in r["title"] for r in filtered)

    def test_filter_case_insensitive(self):
        """Test that filter is case-insensitive."""
        results = [
            {"title": "FMRI Study"},
            {"title": "fmri study"},
            {"title": "Ephys data"},
        ]

        filtered = apply_negative_filter(results, ["fMRI"])

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Ephys data"

    def test_filter_empty_terms(self):
        """Test filter with empty negative terms."""
        results = [{"title": "Test 1"}, {"title": "Test 2"}]

        filtered = apply_negative_filter(results, [])

        assert filtered == results

    def test_filter_custom_fields(self):
        """Test filter with custom text fields."""
        results = [
            {"name": "Visual", "content": "fMRI data"},
            {"name": "Ephys", "content": "Spike sorting"},
        ]

        filtered = apply_negative_filter(
            results,
            ["fMRI"],
            text_fields=["content"],
        )

        assert len(filtered) == 1
        assert filtered[0]["name"] == "Ephys"


class TestParsedTerm:
    """Tests for ParsedTerm dataclass."""

    def test_creation(self):
        """Test term creation."""
        term = ParsedTerm(
            term="ephys",
            constraint_type=ConstraintType.POSITIVE,
        )

        assert term.term == "ephys"
        assert term.constraint_type == ConstraintType.POSITIVE
        assert term.source == "explicit"
        assert term.confidence == 1.0

    def test_implicit_term(self):
        """Test implicit term creation."""
        term = ParsedTerm(
            term="ephys",
            constraint_type=ConstraintType.REQUIRED,
            source="implicit",
            confidence=0.9,
        )

        assert term.source == "implicit"
        assert term.confidence == 0.9
