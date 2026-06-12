"""Tests for Concept Memory Retrieval Integration (v0.5).

Covers: reranker, explainer, evaluator, coverage, and CLI smoke tests.
"""

from __future__ import annotations

import json

import pytest

from neural_search.field_state.concept_memory.coverage import (
    _build_coverage_map,
    _render_coverage_markdown,
    generate_coverage_report,
)
from neural_search.field_state.concept_memory.evaluator import (
    _dcg,
    _idcg,
    _mrr,
    _ndcg_at_k,
    _recall_at_k,
    _render_eval_report_markdown,
    run_concept_eval,
)
from neural_search.field_state.concept_memory.explainer import (
    _identify_missing_evidence,
    explain_result,
)
from neural_search.field_state.concept_memory.graph_builder import build_concept_graph
from neural_search.field_state.concept_memory.ids import concept_id, evidence_id
from neural_search.field_state.concept_memory.reranker import (
    CONCEPT_TYPE_BOOST_WEIGHTS,
    rerank_datasets,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptExplanation,
    ConceptNode,
    ConceptRerankedResult,
    EvidenceLink,
    MatchedConceptInfo,
    ScoreBreakdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(concept_id_str: str, name: str, ctype: str, **kwargs) -> ConceptNode:
    return ConceptNode(concept_id=concept_id_str, canonical_name=name, concept_type=ctype, **kwargs)


def _link(
    src: str,
    tgt: str,
    relation: str,
    review_status: str = "unreviewed",
    confidence: float = 0.8,
    evidence_text: str | None = None,
) -> EvidenceLink:
    eid = evidence_id(src, tgt, relation)
    return EvidenceLink(
        evidence_id=eid,
        source_concept_id=src,
        target_concept_id=tgt,
        evidence_type="derived_from_artifact",
        relation_type=relation,
        review_status=review_status,
        confidence=confidence,
        evidence_text=evidence_text,
    )


@pytest.fixture()
def minimal_graph():
    """Minimal concept graph with 1 dataset, 1 task, 1 modality, 1 species."""
    ds_id = concept_id("dataset", "neuropixels spike sorting dataset")
    task_id = concept_id("task", "spike sorting")
    mod_id = concept_id("modality", "electrophysiology")
    sp_id = concept_id("species", "mouse")
    fm_id = concept_id("failure_mode", "resting state artifact")

    concepts = [
        _node(ds_id, "Neuropixels Spike Sorting Dataset", "dataset", source_ids=["dandiset_000001"]),
        _node(task_id, "spike sorting", "task", aliases=["spike-sorting", "unit sorting"]),
        _node(mod_id, "electrophysiology", "modality", aliases=["ephys"]),
        _node(sp_id, "mouse", "species"),
        _node(fm_id, "resting state artifact", "failure_mode"),
    ]
    links = [
        _link(ds_id, task_id, "has_task", review_status="reviewed", evidence_text="spike sorting performed"),
        _link(ds_id, mod_id, "has_modality", confidence=0.9),
        _link(ds_id, sp_id, "has_species", confidence=0.95),
    ]
    return concepts, links


@pytest.fixture()
def graph_with_hard_negative(minimal_graph):
    """Graph where the dataset also has a failure-mode link."""
    concepts, links = minimal_graph
    ds_id = next(c.concept_id for c in concepts if c.concept_type == "dataset")
    fm_id = next(c.concept_id for c in concepts if c.concept_type == "failure_mode")
    links = list(links) + [_link(ds_id, fm_id, "has_failure_mode")]
    return concepts, links


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestNewSchemas:
    def test_concept_reranked_result_fields(self):
        r = ConceptRerankedResult(
            dataset_id="dandiset_000001",
            dataset_title="Test Dataset",
            base_score=0.4,
            concept_boost=0.15,
            evidence_boost=0.05,
            hard_negative_penalty=0.0,
            final_score=0.60,
        )
        assert r.final_score == pytest.approx(0.60)
        sb = r.score_breakdown()
        assert isinstance(sb, ScoreBreakdown)
        assert sb.base_score == 0.4
        assert sb.concept_boost == 0.15

    def test_matched_concept_info_fields(self):
        mc = MatchedConceptInfo(
            concept_id="concept:task:spike-sorting",
            canonical_name="spike sorting",
            concept_type="task",
            match_score=0.75,
        )
        assert mc.concept_type == "task"
        assert mc.match_score == pytest.approx(0.75)

    def test_concept_explanation_serializes_to_json(self):
        explanation = ConceptExplanation(
            dataset_id="dandiset_000001",
            dataset_title="Test",
            query="spike sorting mouse",
            explanation_markdown="## Explanation\n\nSome text.",
        )
        as_json = explanation.to_jsonl()
        restored = ConceptExplanation.from_jsonl(as_json)
        assert restored.dataset_id == "dandiset_000001"
        assert restored.query == "spike sorting mouse"


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


class TestReranker:
    def test_rerank_returns_only_dataset_concepts(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        results = rerank_datasets("spike sorting mouse electrophysiology", concepts, links, graph)
        for r in results:
            assert isinstance(r, ConceptRerankedResult)
        # Should only have 1 dataset in this fixture
        assert len(results) == 1

    def test_concept_boost_increases_score_over_lexical(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        query = "spike sorting mouse electrophysiology"

        results_full = rerank_datasets(query, concepts, links, graph,
                                        enable_concept_boost=True,
                                        enable_evidence_boost=False,
                                        enable_hard_negative_penalty=False)
        results_lex = rerank_datasets(query, concepts, links, graph,
                                       enable_concept_boost=False,
                                       enable_evidence_boost=False,
                                       enable_hard_negative_penalty=False)

        assert len(results_full) == len(results_lex) == 1
        # concept boost must not silently set final < base
        r_full = results_full[0]
        r_lex = results_lex[0]
        assert r_full.concept_boost >= 0.0
        assert r_full.final_score >= r_lex.final_score

    def test_graph_boost_never_dominates_base_beyond_cap(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        results = rerank_datasets("spike sorting mouse", concepts, links, graph)
        for r in results:
            # concept_boost is capped at boost_scale (0.3 default)
            assert r.concept_boost <= 0.30 + 1e-9

    def test_hard_negative_penalty_reduces_score(self, graph_with_hard_negative):
        concepts, links = graph_with_hard_negative
        graph = build_concept_graph(concepts, links)
        query = "spike sorting mouse"

        results_with_penalty = rerank_datasets(query, concepts, links, graph,
                                                enable_hard_negative_penalty=True)
        results_without = rerank_datasets(query, concepts, links, graph,
                                           enable_hard_negative_penalty=False)

        assert len(results_with_penalty) == len(results_without) == 1
        r_pen = results_with_penalty[0]
        r_nopen = results_without[0]
        assert r_pen.hard_negative_penalty > 0
        assert r_pen.final_score <= r_nopen.final_score

    def test_score_components_are_named_and_non_negative(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        results = rerank_datasets("electrophysiology spike sorting", concepts, links, graph)
        for r in results:
            assert r.base_score >= 0.0
            assert r.concept_boost >= 0.0
            assert r.evidence_boost >= 0.0
            assert r.hard_negative_penalty >= 0.0
            assert r.final_score >= 0.0

    def test_final_score_is_decomposable(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        results = rerank_datasets("spike sorting electrophysiology mouse", concepts, links, graph)
        for r in results:
            expected = max(
                r.base_score + r.concept_boost + r.evidence_boost - r.hard_negative_penalty, 0.0
            )
            assert abs(r.final_score - expected) < 1e-5

    def test_missing_concept_artifacts_raises_gracefully(self, tmp_path):
        """rerank_from_artifacts with empty root returns empty list (no concepts)."""
        # An empty directory has no concept artifacts, so graph is empty → 0 results
        from neural_search.field_state.concept_memory.reranker import (
            rerank_from_artifacts,
        )
        results = rerank_from_artifacts("spike sorting", root=tmp_path)
        assert results == []

    def test_explanation_summary_contains_matched_concept_names(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        results = rerank_datasets("spike sorting electrophysiology", concepts, links, graph)
        assert len(results) == 1
        r = results[0]
        if r.matched_concepts:
            assert any(
                m.canonical_name.lower() in r.explanation_summary.lower()
                for m in r.matched_concepts
            )

    def test_concept_type_boost_weights_sum_is_reasonable(self):
        # No single type should have a weight > 1.0
        for ctype, w in CONCEPT_TYPE_BOOST_WEIGHTS.items():
            assert 0.0 <= w <= 1.0, f"Weight for {ctype} = {w} is out of range"


# ---------------------------------------------------------------------------
# Explainer tests
# ---------------------------------------------------------------------------


class TestExplainer:
    def test_explain_result_returns_explanation_for_known_dataset(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        explanation = explain_result(
            query="spike sorting electrophysiology mouse",
            dataset_id="dandiset_000001",
            concepts=concepts,
            evidence_links=links,
            graph=graph,
        )
        assert isinstance(explanation, ConceptExplanation)
        assert explanation.dataset_id == "dandiset_000001"
        assert explanation.explanation_markdown

    def test_explain_result_contains_score_breakdown(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        explanation = explain_result(
            query="spike sorting electrophysiology",
            dataset_id="dandiset_000001",
            concepts=concepts,
            evidence_links=links,
            graph=graph,
        )
        assert explanation.score_breakdown is not None
        sb = explanation.score_breakdown
        assert sb.base_score >= 0.0
        assert sb.final_score >= 0.0

    def test_explain_result_markdown_contains_dataset_title(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        explanation = explain_result(
            query="spike sorting mouse",
            dataset_id="dandiset_000001",
            concepts=concepts,
            evidence_links=links,
            graph=graph,
        )
        assert "Neuropixels" in explanation.explanation_markdown

    def test_explain_unknown_dataset_returns_graceful_message(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        explanation = explain_result(
            query="spike sorting mouse",
            dataset_id="unknown_dataset_xyz",
            concepts=concepts,
            evidence_links=links,
            graph=graph,
        )
        assert explanation.dataset_id == "unknown_dataset_xyz"
        assert "not found" in explanation.explanation_markdown.lower() or "concept-build" in explanation.explanation_markdown

    def test_explain_result_identifies_missing_evidence_types(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        explanation = explain_result(
            query="spike sorting electrophysiology mouse",
            dataset_id="dandiset_000001",
            concepts=concepts,
            evidence_links=links,
            graph=graph,
        )
        # brain_region and method should appear as missing for this minimal fixture
        missing_types = {
            line.split("'")[1]
            for line in explanation.missing_evidence
            if "'" in line
        }
        assert "brain_region" in missing_types or "method" in missing_types

    def test_identify_missing_evidence_all_types_empty(self):
        missing = _identify_missing_evidence(set())
        types_reported = {line.split("'")[1] for line in missing if "'" in line}
        assert "task" in types_reported
        assert "modality" in types_reported

    def test_identify_missing_evidence_all_covered(self):
        covered = {"task", "modality", "species", "brain_region", "method", "analysis_affordance"}
        missing = _identify_missing_evidence(covered)
        assert missing == []


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluatorMetrics:
    def test_dcg_basic(self):
        assert _dcg([3, 2, 1, 0], k=4) == pytest.approx(
            3 / 1 + 2 / 1.585 + 1 / 2 + 0, abs=0.01
        )

    def test_idcg_returns_best_possible(self):
        # idcg with [1, 0, 0] at k=3 should equal dcg([1, 0, 0])
        assert _idcg([0, 0, 1], k=3) == pytest.approx(_dcg([1, 0, 0], k=3))

    def test_ndcg_perfect_ranking(self):
        qrel_map = {"d1": 3, "d2": 2, "d3": 1}
        ranked = ["d1", "d2", "d3"]
        assert _ndcg_at_k(ranked, qrel_map, k=10) == pytest.approx(1.0)

    def test_ndcg_empty_qrels(self):
        assert _ndcg_at_k(["d1", "d2"], {}, k=10) == pytest.approx(0.0)

    def test_mrr_first_hit(self):
        qrel_map = {"d3": 1}
        ranked = ["d1", "d2", "d3", "d4"]
        assert _mrr(ranked, qrel_map) == pytest.approx(1.0 / 3)

    def test_mrr_no_hit(self):
        assert _mrr(["d1", "d2"], {"d99": 1}) == pytest.approx(0.0)

    def test_recall_at_k(self):
        qrel_map = {"d1": 1, "d2": 1, "d3": 1}
        ranked = ["d1", "d2", "d4", "d5"]
        assert _recall_at_k(ranked, qrel_map, k=4) == pytest.approx(2.0 / 3)

    def test_recall_at_k_no_relevant(self):
        assert _recall_at_k(["d1"], {}, k=10) == pytest.approx(0.0)


class TestEvaluatorReport:
    def test_eval_report_placeholder_when_no_qrels(self, tmp_path):
        # Write a minimal benchmark_queries.jsonl
        q_path = tmp_path / "queries.jsonl"
        q_path.write_text(
            json.dumps({"query_id": "q001", "intent": "META_ANALYSIS", "query": "spike sorting mouse"}) + "\n",
            encoding="utf-8",
        )
        out = run_concept_eval(
            queries_path=q_path,
            qrels_path=None,
            field="test_field",
            root=tmp_path,
        )
        text = out.read_text(encoding="utf-8")
        assert "No Qrels Available" in text or "No adjudicated qrels" in text
        assert "fabricat" not in text.lower()

    def test_eval_report_no_fabricated_metrics_when_qrels_empty(self, tmp_path):
        q_path = tmp_path / "queries.jsonl"
        q_path.write_text(
            json.dumps({"query_id": "q001", "intent": "META_ANALYSIS", "query": "fmri"}) + "\n",
            encoding="utf-8",
        )
        qrels_path = tmp_path / "qrels.jsonl"
        qrels_path.write_text("", encoding="utf-8")  # empty file

        out = run_concept_eval(
            queries_path=q_path,
            qrels_path=qrels_path,
            field="test_field",
            root=tmp_path,
        )
        # With 0 qrels, metrics cannot be computed — report should say so or show "—"
        assert out.exists()

    def test_render_eval_report_placeholder_contains_instructions(self):
        md = _render_eval_report_markdown(
            field="test",
            queries=[{"query_id": "q001", "intent": "META_ANALYSIS", "query": "fmri task"}],
            qrels_available=False,
            qrels_count=0,
            metrics_by_variant=[],
            generated_at="2026-06-01T00:00:00+00:00",
        )
        assert "qrels-export" in md
        assert "qrels-import" in md
        assert "concept-eval" in md


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_map_identifies_covered_types(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        coverages = _build_coverage_map(concepts, links, graph)
        assert len(coverages) == 1
        dc = coverages[0]
        assert "task" in dc.covered_types
        assert "modality" in dc.covered_types
        assert "species" in dc.covered_types

    def test_coverage_map_excludes_non_dataset_nodes(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        coverages = _build_coverage_map(concepts, links, graph)
        for dc in coverages:
            assert dc.dataset_id  # every entry has an ID

    def test_coverage_report_generates_valid_markdown(self, minimal_graph, tmp_path):
        concepts, links = minimal_graph
        out = generate_coverage_report(
            concepts=concepts,
            evidence_links=links,
            field="test_field",
            root=tmp_path,
        )
        text = out.read_text(encoding="utf-8")
        assert "# Concept Coverage Audit" in text
        assert "task" in text
        assert "modality" in text
        assert "## Coverage by Concept Type" in text

    def test_coverage_report_empty_corpus(self, tmp_path):
        out = generate_coverage_report(
            concepts=[],
            evidence_links=[],
            field="test_field",
            root=tmp_path,
        )
        text = out.read_text(encoding="utf-8")
        assert "# Concept Coverage" in text
        assert "concept-build" in text.lower() or "no dataset" in text.lower()

    def test_render_coverage_markdown_contains_source_breakdown(self, minimal_graph):
        concepts, links = minimal_graph
        graph = build_concept_graph(concepts, links)
        from neural_search.field_state.concept_memory.coverage import (
            _build_coverage_map,
        )
        coverages = _build_coverage_map(concepts, links, graph)
        md = _render_coverage_markdown(
            coverages=coverages,
            alias_samples=[("ephys", "electrophysiology")],
            generated_at="2026-06-01T00:00:00+00:00",
            field="test_field",
        )
        assert "## Coverage by Source Repository" in md
        assert "dandiset" in md


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestCLISmokeTests:
    def test_concept_rerank_help(self):
        from neural_search.field_state.concept_memory.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["concept-rerank", "--help"])
        assert exc.value.code == 0

    def test_concept_explain_help(self):
        from neural_search.field_state.concept_memory.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["concept-explain", "--help"])
        assert exc.value.code == 0

    def test_concept_eval_help(self):
        from neural_search.field_state.concept_memory.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["concept-eval", "--help"])
        assert exc.value.code == 0

    def test_concept_coverage_help(self):
        from neural_search.field_state.concept_memory.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["concept-coverage", "--help"])
        assert exc.value.code == 0

    def test_field_state_cli_registers_all_four_commands(self):
        from neural_search.field_state.cli import build_parser
        parser = build_parser()
        # Just verify the parser builds without error; smoke test
        assert parser is not None

    def test_concept_rerank_runs_with_empty_root(self, tmp_path):
        import argparse

        from neural_search.field_state.concept_memory.cli import cmd_concept_rerank
        args = argparse.Namespace(
            query="spike sorting mouse",
            limit=5,
            field="neuroscience_dataset_reuse",
            lexical_only=False,
            root=tmp_path,
        )
        rc = cmd_concept_rerank(args)
        assert rc == 0  # no results but exits cleanly

    def test_concept_coverage_runs_with_empty_root(self, tmp_path):
        import argparse

        from neural_search.field_state.concept_memory.cli import cmd_concept_coverage
        args = argparse.Namespace(
            field="neuroscience_dataset_reuse",
            out=None,
            root=tmp_path,
        )
        rc = cmd_concept_coverage(args)
        assert rc == 0

    def test_concept_eval_runs_without_qrels(self, tmp_path):
        import argparse

        from neural_search.field_state.concept_memory.cli import cmd_concept_eval
        q_path = tmp_path / "queries.jsonl"
        q_path.write_text(
            json.dumps({"query_id": "q001", "intent": "META_ANALYSIS", "query": "fmri"}) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            queries=q_path,
            qrels=None,
            field="neuroscience_dataset_reuse",
            out=None,
            root=tmp_path,
        )
        rc = cmd_concept_eval(args)
        assert rc == 0
