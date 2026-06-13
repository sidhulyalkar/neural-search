"""Tests for the Concept Memory retrieval ablation harness."""

from __future__ import annotations

import json
from pathlib import Path

from neural_search.field_state.concept_memory.ablation import (
    DEFAULT_JSON_OUT,
    _hard_negative_violation_rate,
    run_ablation,
)
from neural_search.field_state.concept_memory.cli import main
from neural_search.field_state.concept_memory.graph_builder import (
    write_concept_artifacts,
)
from neural_search.field_state.concept_memory.ids import evidence_id
from neural_search.field_state.concept_memory.schema import ConceptNode, EvidenceLink


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _concept_memory_fixture(root: Path) -> tuple[Path, Path, Path]:
    concepts = [
        ConceptNode(
            concept_id="concept:dataset:ds1",
            canonical_name="Mouse Neuropixels visual cortex spike sorting dataset",
            concept_type="dataset",
            description="Extracellular Neuropixels recordings for spike sorting in mouse VISp.",
            source_ids=["ds1"],
            evidence_count=2,
            review_status="reviewed",
            confidence=0.9,
        ),
        ConceptNode(
            concept_id="concept:dataset:ds2",
            canonical_name="Human resting state fMRI dataset",
            concept_type="dataset",
            description="Resting state fMRI scans from human participants.",
            source_ids=["ds2"],
            evidence_count=1,
            review_status="reviewed",
            confidence=0.8,
        ),
        ConceptNode(
            concept_id="concept:modality:neuropixels",
            canonical_name="neuropixels",
            concept_type="modality",
            description="High-density electrophysiology probe recordings.",
            evidence_count=1,
            review_status="reviewed",
            confidence=0.9,
        ),
        ConceptNode(
            concept_id="concept:task:spike-sorting",
            canonical_name="spike sorting",
            concept_type="task",
            description="Identifying units from extracellular spike waveforms.",
            evidence_count=1,
            review_status="reviewed",
            confidence=0.9,
        ),
        ConceptNode(
            concept_id="concept:modality:fmri",
            canonical_name="fmri",
            concept_type="modality",
            description="Functional magnetic resonance imaging.",
            evidence_count=1,
            review_status="reviewed",
            confidence=0.8,
        ),
    ]
    links = [
        EvidenceLink(
            evidence_id=evidence_id(
                "concept:dataset:ds1",
                "concept:modality:neuropixels",
                "has_modality",
            ),
            source_concept_id="concept:dataset:ds1",
            target_concept_id="concept:modality:neuropixels",
            evidence_type="derived_from_artifact",
            relation_type="has_modality",
            evidence_text="Dataset ds1 lists Neuropixels as a structured modality.",
            confidence=0.9,
            review_status="reviewed",
            source_repository="fixture",
            source_record_id="ds1",
            source_field="modalities",
            extractor_name="test_fixture",
            extractor_version="1",
            metadata={"evidence_strength": "moderate"},
        ),
        EvidenceLink(
            evidence_id=evidence_id(
                "concept:dataset:ds1",
                "concept:task:spike-sorting",
                "has_task",
            ),
            source_concept_id="concept:dataset:ds1",
            target_concept_id="concept:task:spike-sorting",
            evidence_type="derived_from_artifact",
            relation_type="has_task",
            evidence_text="Dataset ds1 lists spike sorting as a task.",
            confidence=0.9,
            review_status="reviewed",
            source_repository="fixture",
            source_record_id="ds1",
            source_field="tasks",
            extractor_name="test_fixture",
            extractor_version="1",
            metadata={"evidence_strength": "moderate"},
        ),
        EvidenceLink(
            evidence_id=evidence_id("concept:dataset:ds2", "concept:modality:fmri", "has_modality"),
            source_concept_id="concept:dataset:ds2",
            target_concept_id="concept:modality:fmri",
            evidence_type="derived_from_artifact",
            relation_type="has_modality",
            evidence_text="Dataset ds2 lists fMRI as a structured modality.",
            confidence=0.8,
            review_status="reviewed",
            source_repository="fixture",
            source_record_id="ds2",
            source_field="modalities",
            extractor_name="test_fixture",
            extractor_version="1",
            metadata={"evidence_strength": "weak"},
        ),
    ]
    write_concept_artifacts(concepts, links, root=root, deterministic=True)

    queries = root / "queries.jsonl"
    qrels = root / "qrels.jsonl"
    corpus = root / "corpus.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "query_text": "neuropixels spike sorting",
                "intent": "REPLICATION",
            },
            {
                "query_id": "q2",
                "query_text": "resting state fmri",
                "intent": "DISCOVERY",
            },
        ],
    )
    _write_jsonl(
        qrels,
        [
            {"query_id": "q1", "record_id": "ds1", "relevance": 3},
            {"query_id": "q1", "record_id": "ds2", "relevance": 0},
            {"query_id": "q2", "record_id": "ds2", "relevance": 3},
            {"query_id": "q2", "record_id": "ds1", "relevance": 0},
        ],
    )
    _write_jsonl(
        corpus,
        [
            {"source": "fixture", "source_id": "ds1"},
            {"source": "fixture", "source_id": "ds2"},
        ],
    )
    return queries, qrels, corpus


def test_ablation_writes_metrics_and_intent_breakdowns(tmp_path: Path):
    queries, qrels, corpus = _concept_memory_fixture(tmp_path)
    out = tmp_path / DEFAULT_JSON_OUT

    report = run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=qrels,
        corpus_path=corpus,
        out_json=out,
        deterministic=True,
    )

    assert report["status"] == "computed"
    assert report["retrieval_variants"] == [
        "baseline_no_concept",
        "concept_memory_enabled",
        "concept_memory_lexical_only",
        "concept_memory_graph_degree_normalized",
    ]
    for variant in report["retrieval_variants"]:
        assert "ndcg_at_10" in report["metrics_by_variant"][variant]
        assert report["metrics_by_variant"][variant]["evaluated_queries"] == 2
    assert "REPLICATION" in report["per_intent_metrics"]["baseline_no_concept"]
    assert "concept_memory_enabled" in report["metric_deltas_against_baseline"]
    assert "ndcg_at_10_delta_ci95" in report["metric_deltas_against_baseline"]["concept_memory_enabled"]
    assert report["run_metadata"]["input_hashes"]["qrels_sha256"]
    assert out.exists()
    assert out.with_suffix(".md").exists()


def test_ablation_missing_qrels_writes_placeholder_without_metrics(tmp_path: Path):
    queries, _qrels, corpus = _concept_memory_fixture(tmp_path)

    report = run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=tmp_path / "missing_qrels.jsonl",
        corpus_path=corpus,
        out_json=tmp_path / DEFAULT_JSON_OUT,
        deterministic=True,
    )

    assert report["status"] == "pending_qrels"
    assert report["metrics_by_variant"] == {}
    assert any("Qrels file not found" in warning for warning in report["warnings"])
    assert any("No qrels loaded" in warning for warning in report["warnings"])


def test_ablation_empty_and_malformed_qrels_do_not_emit_fake_metrics(tmp_path: Path):
    queries, _qrels, corpus = _concept_memory_fixture(tmp_path)
    empty_qrels = tmp_path / "empty_qrels.jsonl"
    empty_qrels.write_text("", encoding="utf-8")

    empty_report = run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=empty_qrels,
        corpus_path=corpus,
        out_json=tmp_path / "empty.json",
        deterministic=True,
    )

    malformed_qrels = tmp_path / "malformed_qrels.jsonl"
    malformed_qrels.write_text("{not json}\n", encoding="utf-8")
    malformed_report = run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=malformed_qrels,
        corpus_path=corpus,
        out_json=tmp_path / "malformed.json",
        deterministic=True,
    )

    assert empty_report["status"] == "pending_qrels"
    assert empty_report["metrics_by_variant"] == {}
    assert malformed_report["status"] == "pending_qrels"
    assert malformed_report["metrics_by_variant"] == {}
    assert any("malformed JSONL" in warning for warning in malformed_report["warnings"])


def test_ablation_deterministic_output_is_byte_stable(tmp_path: Path):
    queries, qrels, corpus = _concept_memory_fixture(tmp_path)
    out = tmp_path / DEFAULT_JSON_OUT

    run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=qrels,
        corpus_path=corpus,
        out_json=out,
        deterministic=True,
    )
    first = out.read_bytes()
    run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=qrels,
        corpus_path=corpus,
        out_json=out,
        deterministic=True,
    )
    second = out.read_bytes()

    assert first == second


def test_ablation_report_claim_safety_wording(tmp_path: Path):
    queries, qrels, corpus = _concept_memory_fixture(tmp_path)
    out = tmp_path / DEFAULT_JSON_OUT

    run_ablation(
        root=tmp_path,
        queries_path=queries,
        qrels_path=qrels,
        corpus_path=corpus,
        out_json=out,
        deterministic=True,
    )
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")

    assert "Claim Safety Notice" in markdown
    assert "does not establish general retrieval improvement" in markdown
    assert "Concept Memory improves retrieval" not in markdown
    assert "Metadata-derived links are not reviewed scientific evidence" in markdown


def test_hard_negative_violation_rate_counts_top_k_negatives():
    qrel = {"relevant": 3, "negative": 0}

    assert _hard_negative_violation_rate(qrel, ["negative", "relevant"], k=10) == 1.0
    assert _hard_negative_violation_rate(qrel, ["relevant"], k=10) == 0.0


def test_ablation_cli_command_writes_reports(tmp_path: Path):
    queries, qrels, corpus = _concept_memory_fixture(tmp_path)
    out = tmp_path / "reports" / "eval" / "ablation.json"

    code = main([
        "--root",
        str(tmp_path),
        "concept-ablate-retrieval",
        "--queries",
        str(queries),
        "--qrels",
        str(qrels),
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--deterministic",
    ])

    assert code == 0
    assert out.exists()
    assert out.with_suffix(".md").exists()
