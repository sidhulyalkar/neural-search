from neural_search.embeddings import (
    HashingEmbeddingProvider,
    build_field_embedding_records,
    write_field_embedding_cache,
)
from neural_search.graph import build_graph_from_records, write_graph_json
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
)
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)
from neural_search.search import search_datasets


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.6.0",
    )


def test_search_adds_graph_score_when_graph_config_is_enabled(tmp_path):
    normalized = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("demo", "GRAPHY"),
        source="demo",
        source_id="GRAPHY",
        title="Mouse reversal learning Neuropixels",
        tasks=[_label("task", "reversal_learning")],
        modalities=[_label("modality", "neuropixels")],
        brain_regions=[_label("brain_region", "orbitofrontal_cortex")],
        species=[_label("species", "mouse")],
        analysis_affordances=[
            AnalysisAffordance(
                analysis_id="decoding",
                support_level="high",
                confidence=0.9,
                required_fields_present=["spike_times"],
                evidence=["spike_times"],
            )
        ],
        linked_papers=[make_paper_id("demo", "P1")],
    )
    paper = NormalizedPaperRecord(
        paper_id=make_paper_id("demo", "P1"),
        source="demo",
        source_id="P1",
        title="Reversal paper",
        linked_datasets=[normalized.dataset_id],
    )
    graph_path = write_graph_json(build_graph_from_records([normalized], [paper]), tmp_path / "graph.json")

    response = search_datasets(
        "mouse reversal learning neuropixels decoding",
        datasets=[
            {
                "dataset": {
                    "id": "GRAPHY",
                    "source": "demo",
                    "source_id": "GRAPHY",
                    "title": "Mouse reversal learning Neuropixels",
                    "description": "OFC reversal learning with event timestamps.",
                    "species": ["mouse"],
                    "modalities": ["neuropixels"],
                    "brain_regions": ["orbitofrontal_cortex"],
                    "tasks": ["reversal_learning"],
                    "behaviors": ["choice"],
                    "data_standards": ["NWB"],
                    "has_behavior": True,
                    "has_trials": True,
                    "license": "CC-BY-4.0",
                    "metadata_json": {},
                },
                "card": {
                    "dataset_id": "GRAPHY",
                    "summary": "Reversal learning dataset.",
                    "scientific_labels": {},
                    "analysis_readiness": {"score": 90},
                    "missing_fields": [],
                    "suggested_analyses": ["decoding"],
                    "provenance": {},
                },
            }
        ],
        retrieval_config={"graph": {"enabled": True, "path": str(graph_path)}},
    )

    breakdown = response.results[0].score_breakdown
    assert "graph_score" in breakdown
    assert breakdown["graph_score"] > 0
    graph_context = response.results[0].graph_context or {}
    assert graph_context["requirement_matches"]["modality"]
    assert any(
        "Graph requirements matched" in reason
        for reason in response.results[0].why_matched
    )


def test_search_adds_field_semantic_score_when_embedding_config_is_enabled(tmp_path):
    normalized = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("demo", "FIELDY"),
        source="demo",
        source_id="FIELDY",
        title="Mouse go nogo calcium imaging",
        description="Lick and reward omission events in mPFC.",
        tasks=[_label("task", "go_nogo")],
        behavioral_events=[_label("behavioral_event", "lick")],
        modalities=[_label("modality", "calcium_imaging")],
        brain_regions=[_label("brain_region", "mPFC")],
    )
    records = build_field_embedding_records(
        [normalized],
        HashingEmbeddingProvider(dimensions=16),
        created_at="2026-05-24T00:00:00+00:00",
    )
    cache_path = write_field_embedding_cache(records, tmp_path / "field_embeddings.jsonl")

    response = search_datasets(
        "go nogo calcium lick omission",
        datasets=[
            {
                "dataset": {
                    "id": "FIELDY",
                    "source": "demo",
                    "source_id": "FIELDY",
                    "title": "Mouse go nogo calcium imaging",
                    "description": "Lick and reward omission events in mPFC.",
                    "species": ["mouse"],
                    "modalities": ["calcium_imaging"],
                    "brain_regions": ["mPFC"],
                    "tasks": ["go_nogo"],
                    "behaviors": ["lick"],
                    "data_standards": ["NWB"],
                    "has_behavior": True,
                    "has_trials": True,
                    "license": "CC-BY-4.0",
                    "metadata_json": {},
                },
                "card": {
                    "dataset_id": "FIELDY",
                    "summary": "Go NoGo calcium dataset.",
                    "scientific_labels": {},
                    "analysis_readiness": {"score": 90},
                    "missing_fields": [],
                    "suggested_analyses": [],
                    "provenance": {},
                },
            }
        ],
        retrieval_config={"field_embeddings": {"enabled": True, "path": str(cache_path)}},
    )

    breakdown = response.results[0].score_breakdown
    assert "field_semantic_score" in breakdown
    assert breakdown["field_semantic_score"] > 0
