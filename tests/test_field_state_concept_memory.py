"""Tests for the Graph-Indexed Concept Memory module.

20 tests covering: schema, IDs, normalization, loaders, graph builder,
basis generation, retrieval, reports, and Obsidian export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_search.field_state.concept_memory import (
    graph_builder as concept_graph_builder,
)
from neural_search.field_state.concept_memory.basis import generate_all_bases
from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    get_concept_neighborhood,
    write_concept_artifacts,
)
from neural_search.field_state.concept_memory.ids import concept_id, evidence_id
from neural_search.field_state.concept_memory.loaders import (
    load_corpus,
    load_field_state_artifacts,
    load_obsidian_notes,
)
from neural_search.field_state.concept_memory.manifest import read_manifest
from neural_search.field_state.concept_memory.normalize import (
    is_same_concept,
    normalize_concept_name,
)
from neural_search.field_state.concept_memory.obsidian_export import (
    export_concept_memory_to_obsidian,
)
from neural_search.field_state.concept_memory.reports import generate_all_reports
from neural_search.field_state.concept_memory.retrieval import search_concepts
from neural_search.field_state.concept_memory.schema import (
    ConceptBasis,
    ConceptNode,
    EvidenceLink,
)
from neural_search.field_state.concept_memory.validators import validate_concept_memory
from neural_search.field_state.obsidian.templates import HUMAN_BEGIN

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(concept_id_str: str, name: str, ctype: str, **kwargs) -> ConceptNode:
    """Helper to create a ConceptNode."""
    return ConceptNode(
        concept_id=concept_id_str,
        canonical_name=name,
        concept_type=ctype,
        **kwargs,
    )


def _link(
    src: str,
    tgt: str,
    relation: str,
    review_status: str = "unreviewed",
) -> EvidenceLink:
    """Helper to create an EvidenceLink."""
    eid = evidence_id(src, tgt, relation)
    return EvidenceLink(
        evidence_id=eid,
        source_concept_id=src,
        target_concept_id=tgt,
        evidence_type="derived_from_artifact",
        relation_type=relation,
        review_status=review_status,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_concept_node_schema_validates_required_fields():
    """ConceptNode validates required fields and rejects invalid ones."""
    node = ConceptNode(
        concept_id="concept:method:spike-sorting",
        canonical_name="spike sorting",
        concept_type="method",
    )
    assert node.concept_id == "concept:method:spike-sorting"
    assert node.canonical_name == "spike sorting"
    assert node.concept_type == "method"

    with pytest.raises(ValidationError):
        ConceptNode(
            concept_id="",
            canonical_name="spike sorting",
            concept_type="method",
        )

    with pytest.raises(ValidationError):
        ConceptNode(
            concept_id="concept:method:spike-sorting",
            canonical_name="spike sorting",
            concept_type="not_a_valid_type",
        )


def test_evidence_link_schema_validates_required_fields():
    """EvidenceLink validates required fields and rejects invalid relation_type."""
    link = EvidenceLink(
        evidence_id="evidence:abc:uses_method:123456",
        source_concept_id="concept:dataset:test",
        target_concept_id="concept:method:spike-sorting",
        evidence_type="derived_from_artifact",
        relation_type="uses_method",
    )
    assert link.evidence_id == "evidence:abc:uses_method:123456"
    assert link.relation_type == "uses_method"

    with pytest.raises(ValidationError):
        EvidenceLink(
            evidence_id="evidence:xyz:bad:000000",
            source_concept_id="concept:dataset:test",
            evidence_type="derived_from_artifact",
            relation_type="not_a_valid_relation",
        )


def test_concept_basis_schema_validates_evidence_strength():
    """ConceptBasis validates evidence_strength and rejects invalid values."""
    basis = ConceptBasis(
        concept_id="concept:method:spike-sorting",
        canonical_name="spike sorting",
        concept_type="method",
        evidence_strength="moderate",
    )
    assert basis.evidence_strength == "moderate"

    with pytest.raises(ValidationError):
        ConceptBasis(
            concept_id="concept:method:spike-sorting",
            canonical_name="spike sorting",
            concept_type="method",
            evidence_strength="amazing",
        )


# ---------------------------------------------------------------------------
# ID tests
# ---------------------------------------------------------------------------


def test_stable_concept_id_generation():
    """concept_id produces stable, deterministic IDs."""
    cid1 = concept_id("method", "Spike Sorting")
    assert cid1 == "concept:method:spike-sorting"

    cid2 = concept_id("modality", "Neuropixels")
    assert cid2 == "concept:modality:neuropixels"

    # Idempotent — calling again gives the same result
    assert concept_id("method", "Spike Sorting") == cid1
    assert concept_id("modality", "Neuropixels") == cid2


def test_stable_evidence_id_generation():
    """evidence_id produces deterministic IDs that start with 'evidence:'."""
    src = "concept:method:spike-sorting"
    tgt = "concept:dataset:test"
    relation = "uses_method"

    eid1 = evidence_id(src, tgt, relation)
    eid2 = evidence_id(src, tgt, relation)

    assert eid1 == eid2
    assert eid1.startswith("evidence:")


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_alias_normalization_modalities():
    """normalize_concept_name returns canonical forms for known aliases."""
    assert normalize_concept_name("Neuropixels") == "neuropixels"
    assert normalize_concept_name("two-photon calcium imaging") == "calcium_imaging"
    assert normalize_concept_name("fMRI") == "fmri"


def test_is_same_concept_with_aliases():
    """is_same_concept returns True for aliased names and False for distinct ones."""
    assert is_same_concept("DLC", "DeepLabCut")
    assert not is_same_concept("hippocampus", "cortex")


def test_dangerous_aliases_do_not_false_merge():
    """Ambiguous short labels should only merge when explicitly aliased."""
    assert normalize_concept_name("AP") == "ap"
    assert normalize_concept_name("action potential") == "action_potential"
    assert normalize_concept_name("anterior-posterior") == "anterior-posterior"
    assert not is_same_concept("AP", "action potential")
    assert not is_same_concept("AP", "anterior-posterior")

    assert normalize_concept_name("M1") == "m1"
    assert not is_same_concept("M1", "primary motor cortex")

    assert normalize_concept_name("VISp") == "visp"
    assert not is_same_concept("VISp", "visual cortex")

    assert normalize_concept_name("DLC coating") == "dlc_coating"
    assert not is_same_concept("DLC coating", "DeepLabCut")


def test_loader_does_not_infer_mouse_species_from_free_text(tmp_path: Path):
    """Species concepts should come from structured species fields, not titles."""
    corpus_file = tmp_path / "combined_corpus.jsonl"
    corpus_file.write_text(
        json.dumps({
            "source": "example",
            "source_id": "mouse_device",
            "title": "Computer mouse tracking dataset",
            "description": "USB mouse trajectories from a cursor task.",
            "modalities": [],
            "tasks": [],
            "brain_regions": [],
            "species": [],
        }),
        encoding="utf-8",
    )

    result = load_corpus(corpus_path=corpus_file)

    assert not any(
        c.concept_type == "species" and c.canonical_name == "mouse"
        for c in result.concepts
    )


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_field_state_artifacts():
    """load_field_state_artifacts returns a LoadResult without crashing."""
    result = load_field_state_artifacts()

    # May have warnings (e.g. missing optional files) but should not crash
    assert isinstance(result.concepts, list)
    assert isinstance(result.evidence_links, list)
    assert isinstance(result.warnings, list)
    # At least 1 concept should exist given the project artifacts
    assert len(result.concepts) >= 1


def test_load_corpus_tiny_fixture(tmp_path: Path):
    """load_corpus parses a tiny JSONL fixture and produces dataset/modality/task nodes."""
    corpus_file = tmp_path / "combined_corpus.jsonl"
    records = [
        {
            "source": "dandi",
            "source_id": "000001",
            "title": "Test Dataset A",
            "modalities": ["extracellular_ephys"],
            "tasks": ["go_nogo"],
            "brain_regions": ["hippocampus"],
            "species": ["mouse"],
        },
        {
            "source": "dandi",
            "source_id": "000002",
            "title": "Test Dataset B",
            "modalities": ["calcium_imaging"],
            "tasks": [],
            "brain_regions": ["cortex"],
            "species": ["rat"],
        },
        {
            "source": "gin",
            "source_id": "abc123",
            "title": "Test Dataset C",
            "modalities": ["eeg"],
            "tasks": ["visual_discrimination"],
            "brain_regions": [],
            "species": ["human"],
        },
    ]
    corpus_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    result = load_corpus(corpus_path=corpus_file)

    concept_types = {c.concept_type for c in result.concepts}
    assert "dataset" in concept_types
    assert "modality" in concept_types
    assert "task" in concept_types


def test_corpus_links_include_structured_provenance(tmp_path: Path):
    """Dataset-derived links should expose source field and record provenance."""
    corpus_file = tmp_path / "combined_corpus.jsonl"
    corpus_file.write_text(
        json.dumps({
            "source": "dandi",
            "source_id": "000001",
            "title": "Test Dataset A",
            "modalities": ["neuropixels"],
            "tasks": ["go_nogo"],
            "brain_regions": ["VISp"],
            "species": ["mouse"],
        }),
        encoding="utf-8",
    )

    result = load_corpus(corpus_path=corpus_file)
    by_relation = {link.relation_type: link for link in result.evidence_links}

    for relation, source_field in {
        "has_modality": "modalities",
        "has_task": "tasks",
        "has_brain_region": "brain_regions",
        "has_species": "species",
    }.items():
        link = by_relation[relation]
        assert link.source_repository == "dandi"
        assert link.source_record_id == "000001"
        assert link.source_field == source_field
        assert link.extractor_name == "concept_memory_corpus_loader"
        assert link.evidence_text is not None


def test_load_obsidian_notes_from_temp_vault(tmp_path: Path):
    """load_obsidian_notes picks up notes with recognized frontmatter types."""
    field_state_dir = tmp_path / "Field-State"
    field_state_dir.mkdir(parents=True)

    note = field_state_dir / "test_claim.md"
    note.write_text(
        "---\ntype: claim\ntitle: Dense retrieval helps\nfield_state_id: claim_001\n---\n\n# Body text\n",
        encoding="utf-8",
    )

    result = load_obsidian_notes(vault_path=tmp_path)

    assert len(result.concepts) >= 1
    assert any(c.concept_type == "claim" for c in result.concepts)


# ---------------------------------------------------------------------------
# Graph builder tests
# ---------------------------------------------------------------------------


def test_build_concept_nodes():
    """build_concept_graph produces a graph with the correct node and edge count."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    n3 = _node("concept:modality:ephys", "ephys", "modality")

    l1 = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")
    l2 = _link("concept:dataset:test-a", "concept:modality:ephys", "has_modality")

    g = build_concept_graph([n1, n2, n3], [l1, l2])

    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_build_evidence_links_with_invalid_reference():
    """build_concept_graph skips edges that reference non-existent concept IDs."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")

    valid_link = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")
    invalid_link = _link("concept:dataset:test-a", "concept:method:does-not-exist", "uses_method")

    g = build_concept_graph([n1, n2], [valid_link, invalid_link])

    assert g.number_of_edges() == 1


def test_build_graph_json(tmp_path: Path):
    """write_concept_artifacts produces a valid concept_graph.json with nodes and edges."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    l1 = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")

    paths = write_concept_artifacts([n1, n2], [l1], root=tmp_path)

    graph_json_path = paths["concept_graph"]
    assert graph_json_path.exists()

    data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data


def test_deterministic_artifacts_are_byte_stable(tmp_path: Path):
    """Deterministic mode fixes volatile timestamps in generated JSONL artifacts."""
    n1 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    n2 = _node("concept:modality:ephys", "ephys", "modality")
    l1 = _link("concept:dataset:test-a", "concept:modality:ephys", "has_modality")

    first = write_concept_artifacts([n1, n2], [l1], root=tmp_path / "a", deterministic=True)
    second = write_concept_artifacts([n1, n2], [l1], root=tmp_path / "b", deterministic=True)

    assert first["evidence_links"].read_bytes() == second["evidence_links"].read_bytes()
    assert "1970-01-01T00:00:00+00:00" in first["evidence_links"].read_text(encoding="utf-8")


def test_manifest_semantic_hash_ignores_volatile_timestamps(tmp_path: Path):
    """Semantic hashes remain stable when only approved volatile fields differ."""
    n1 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    n2 = _node("concept:modality:ephys", "ephys", "modality")
    link_a = _link("concept:dataset:test-a", "concept:modality:ephys", "has_modality")
    link_b = link_a.model_copy(update={"created_at": "2030-01-01T00:00:00+00:00"})

    paths_a = write_concept_artifacts([n1, n2], [link_a], root=tmp_path / "a")
    paths_b = write_concept_artifacts([n1, n2], [link_b], root=tmp_path / "b")
    manifest_a = read_manifest(tmp_path / "a")
    manifest_b = read_manifest(tmp_path / "b")
    assert manifest_a is not None
    assert manifest_b is not None

    evidence_a = next(item for item in manifest_a["artifacts"] if item["label"] == "evidence_links")
    evidence_b = next(item for item in manifest_b["artifacts"] if item["label"] == "evidence_links")
    assert paths_a["evidence_links"].read_bytes() != paths_b["evidence_links"].read_bytes()
    assert evidence_a["semantic_sha256"] == evidence_b["semantic_sha256"]


def test_graphml_export_failure_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GraphML export failures should be recorded in the manifest and validation warnings."""
    n1 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    n2 = _node("concept:modality:ephys", "ephys", "modality")
    link = _link("concept:dataset:test-a", "concept:modality:ephys", "has_modality")

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("graphml disabled for test")

    monkeypatch.setattr(concept_graph_builder.nx, "write_graphml", _boom)
    write_concept_artifacts([n1, n2], [link], root=tmp_path)
    bases = generate_all_bases([n1, n2], [link])
    result = validate_concept_memory([n1, n2], [link], bases, artifact_root=tmp_path)
    manifest = read_manifest(tmp_path)

    assert manifest is not None
    assert manifest["graphml_export"]["status"] == "failed"
    assert any("GraphML export" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# Basis tests
# ---------------------------------------------------------------------------


def test_create_concept_basis_records():
    """generate_all_bases returns one ConceptBasis per concept with matching concept_id."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    l1 = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")

    bases = generate_all_bases([n1, n2], [l1])

    assert len(bases) == 2
    basis_ids = {b.concept_id for b in bases}
    assert "concept:method:spike-sorting" in basis_ids
    assert "concept:dataset:test-a" in basis_ids


def test_basis_no_evidence_stays_missing_not_inferred():
    """A concept with no links should be marked as missing evidence."""
    node = _node("concept:method:unlinked", "unlinked method", "method")

    basis = generate_all_bases([node], [])[0]

    assert basis.evidence_strength == "none"
    assert basis.missing_count == 1
    assert basis.evidence_links == []
    assert basis.supporting_claim_ids == []
    assert basis.supporting_dataset_ids == []
    assert any("No evidence links found" in note for note in basis.uncertainty_notes)


def test_basis_contradictory_single_link_remains_weak():
    """One reviewed contradiction should not inflate a basis beyond weak support."""
    claim = _node("concept:claim:c1", "Claim one", "claim")
    method = _node("concept:method:m1", "method one", "method")
    link = _link(
        "concept:claim:c1",
        "concept:method:m1",
        "contradicts",
        review_status="reviewed",
    )

    basis_by_id = {
        basis.concept_id: basis
        for basis in generate_all_bases([claim, method], [link])
    }

    assert basis_by_id["concept:claim:c1"].evidence_strength == "none"
    assert basis_by_id["concept:claim:c1"].reviewed_supporting_count == 0
    assert basis_by_id["concept:claim:c1"].reviewed_contradicting_count == 1
    assert basis_by_id["concept:method:m1"].evidence_strength == "none"


def test_basis_metadata_only_remains_weak_metadata_derived():
    """Metadata-only links should not be counted as scientific support."""
    dataset = _node("concept:dataset:ds", "Dataset", "dataset")
    modality = _node("concept:modality:neuropixels", "neuropixels", "modality")
    link = _link("concept:dataset:ds", "concept:modality:neuropixels", "has_modality")

    basis = {
        item.concept_id: item for item in generate_all_bases([dataset, modality], [link])
    }["concept:dataset:ds"]

    assert basis.evidence_strength == "weak"
    assert basis.supporting_count == 0
    assert basis.neutral_or_metadata_count == 1
    assert basis.metadata_evidence_links == [link.evidence_id]


def test_basis_mixed_supporting_and_contradicting_links_are_visible():
    """Mixed evidence should expose both support and contradiction counts."""
    claim = _node("concept:claim:c1", "Claim one", "claim")
    dataset = _node("concept:dataset:ds", "Dataset", "dataset")
    method = _node("concept:method:m1", "Method", "method")
    support = _link(
        "concept:claim:c1",
        "concept:dataset:ds",
        "supports",
        review_status="reviewed",
    )
    contradict = _link(
        "concept:claim:c1",
        "concept:method:m1",
        "contradicts",
        review_status="reviewed",
    )

    basis = {
        item.concept_id: item
        for item in generate_all_bases([claim, dataset, method], [support, contradict])
    }["concept:claim:c1"]

    assert basis.supporting_count == 1
    assert basis.contradicting_count == 1
    assert basis.reviewed_supporting_count == 1
    assert basis.reviewed_contradicting_count == 1
    assert basis.contradicting_evidence_links == [contradict.evidence_id]


def test_basis_multiple_contradictions_do_not_inflate_support():
    """Contradictions alone should not increase positive evidence strength."""
    claim = _node("concept:claim:c1", "Claim one", "claim")
    methods = [
        _node("concept:method:m1", "Method 1", "method"),
        _node("concept:method:m2", "Method 2", "method"),
    ]
    links = [
        _link("concept:claim:c1", method.concept_id, "contradicts", review_status="reviewed")
        for method in methods
    ]

    basis = {
        item.concept_id: item
        for item in generate_all_bases([claim, *methods], links)
    }["concept:claim:c1"]

    assert basis.evidence_strength == "none"
    assert basis.reviewed_supporting_count == 0
    assert basis.reviewed_contradicting_count == 2


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------


def test_text_only_concept_search():
    """search_concepts returns the spike sorting concept as top result for matching query."""
    n1 = _node("concept:method:spike-sorting", "spike sorting method", "method")
    n2 = _node("concept:method:calcium-imaging", "calcium imaging protocol", "method")
    n3 = _node("concept:task:fmri-task", "fMRI task", "task")

    results = search_concepts("spike sorting", [n1, n2, n3], [], limit=5)

    assert len(results) > 0
    assert results[0].concept_id == "concept:method:spike-sorting"


def test_graph_boosted_retrieval_scoring():
    """Reviewed evidence links boost a concept's score above an unreviewed one."""
    reviewed_node = _node(
        "concept:method:reviewed",
        "spike sorting reviewed",
        "method",
        review_status="reviewed",
    )
    unreviewed_node = _node(
        "concept:method:unreviewed",
        "spike sorting unreviewed",
        "method",
        review_status="unreviewed",
    )

    # Create a dataset node to link from
    ds = _node("concept:dataset:ds", "test ds", "dataset")
    reviewed_link = _link(
        "concept:dataset:ds",
        "concept:method:reviewed",
        "uses_method",
        review_status="reviewed",
    )

    results = search_concepts(
        "spike sorting",
        [reviewed_node, unreviewed_node, ds],
        [reviewed_link],
        limit=10,
    )

    result_ids = [r.concept_id for r in results]
    assert "concept:method:reviewed" in result_ids
    # The reviewed concept should appear before (or at same rank as) the unreviewed one
    reviewed_idx = result_ids.index("concept:method:reviewed")
    unreviewed_idx = result_ids.index("concept:method:unreviewed")
    assert results[reviewed_idx].score >= results[unreviewed_idx].score


def test_graph_boost_does_not_retrieve_zero_lexical_matches():
    """Graph connectivity should not surface a concept with no query text match."""
    irrelevant = _node("concept:method:popular", "popular method", "method")
    ds = _node("concept:dataset:ds", "test ds", "dataset")
    link = _link(
        "concept:dataset:ds",
        "concept:method:popular",
        "uses_method",
        review_status="reviewed",
    )
    graph = build_concept_graph([irrelevant, ds], [link])

    results = search_concepts(
        "calcium imaging",
        [irrelevant, ds],
        [link],
        graph=graph,
        limit=10,
    )

    assert results == []


def test_hub_concept_does_not_dominate_more_lexical_match():
    """Degree-normalized graph boost should not beat a stronger lexical match."""
    exact = _node("concept:method:spike-sorting", "spike sorting", "method")
    hub = _node("concept:method:spike", "spike", "method", evidence_count=20)
    datasets = [
        _node(f"concept:dataset:ds-{idx}", f"Dataset {idx}", "dataset")
        for idx in range(12)
    ]
    links = [
        _link(ds.concept_id, hub.concept_id, "uses_method", review_status="reviewed")
        for ds in datasets
    ]
    graph = build_concept_graph([exact, hub, *datasets], links)

    results = search_concepts(
        "spike sorting",
        [exact, hub, *datasets],
        links,
        graph=graph,
        limit=2,
    )

    assert results[0].concept_id == exact.concept_id
    hub_result = next(result for result in results if result.concept_id == hub.concept_id)
    assert hub_result.graph_boost_raw is not None
    assert hub_result.graph_boost_degree_normalized is not None
    assert hub_result.graph_boost_degree_normalized < hub_result.graph_boost_raw


def test_missingness_penalty_is_bounded_and_explained():
    """Sparse concept metadata should produce a visible bounded penalty."""
    sparse = _node("concept:method:spike-sorting", "spike sorting", "method")

    result = search_concepts("spike sorting", [sparse], [], limit=1)[0]

    assert 0.0 < result.missingness_penalty <= 0.10
    assert result.warnings


def test_score_components_combine_deterministically():
    """The exposed score components should explain final score deterministically."""
    concept = _node(
        "concept:method:spike-sorting",
        "spike sorting",
        "method",
        evidence_count=1,
        description="spike sorting method",
        source_ids=["fixture"],
    )

    result = search_concepts("spike sorting", [concept], [], limit=1)[0]

    expected = round(
        (result.lexical_score or 0.0)
        + (result.graph_boost_degree_normalized or 0.0)
        - result.missingness_penalty,
        6,
    )
    assert result.score == expected
    assert result.final_score == expected
    assert result.matched_terms == ["spike", "sorting"]


def test_concept_neighborhood_query():
    """get_concept_neighborhood returns nodes within depth but not beyond."""
    # Build a chain: A -> B -> C -> D -> E
    nodes = {
        "A": _node("concept:method:a", "concept A", "method"),
        "B": _node("concept:method:b", "concept B", "method"),
        "C": _node("concept:method:c", "concept C", "method"),
        "D": _node("concept:method:d", "concept D", "method"),
        "E": _node("concept:method:e", "concept E", "method"),
    }
    links = [
        _link("concept:method:a", "concept:method:b", "mentions"),
        _link("concept:method:b", "concept:method:c", "mentions"),
        _link("concept:method:c", "concept:method:d", "mentions"),
        _link("concept:method:d", "concept:method:e", "mentions"),
    ]
    g = build_concept_graph(list(nodes.values()), links)

    # Query from B with depth=2 (undirected): should include A, B, C, D
    neighborhood = get_concept_neighborhood(g, "concept:method:b", depth=2)
    node_ids = {n["concept_id"] for n in neighborhood["nodes"]}

    assert "concept:method:a" in node_ids
    assert "concept:method:b" in node_ids
    assert "concept:method:c" in node_ids
    assert "concept:method:d" in node_ids
    # E is 3 hops from B — should NOT be included
    assert "concept:method:e" not in node_ids


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------


def test_concept_report_generation(tmp_path: Path):
    """generate_all_reports creates 7 non-empty report files."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    n3 = _node("concept:claim:c1", "Dense retrieval helps", "claim")

    l1 = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")
    l2 = _link("concept:claim:c1", "concept:dataset:test-a", "supports")

    bases = generate_all_bases([n1, n2, n3], [l1, l2])

    paths = generate_all_reports([n1, n2, n3], [l1, l2], bases, root=tmp_path)

    assert len(paths) == 7
    for label, path in paths.items():
        assert path.exists(), f"Report {label} not found at {path}"
        assert path.stat().st_size > 0, f"Report {label} is empty"

    graph_summary = paths["concept_graph_summary"].read_text(encoding="utf-8")
    claim_map = paths["claim_basis_map"].read_text(encoding="utf-8")
    top_concepts = paths["top_concepts"].read_text(encoding="utf-8")
    assert "Claim Safety Notice" in graph_summary
    assert "Most Connected Metadata Concepts" in graph_summary
    assert "Reviewed supporting evidence" in claim_map
    assert "# Metadata-Supported Concepts" in top_concepts


# ---------------------------------------------------------------------------
# Obsidian export tests
# ---------------------------------------------------------------------------


def test_obsidian_concept_export(tmp_path: Path):
    """export_concept_memory_to_obsidian creates the Concepts directory with .md files."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    n2 = _node("concept:dataset:test-a", "Test Dataset A", "dataset")
    l1 = _link("concept:dataset:test-a", "concept:method:spike-sorting", "uses_method")
    bases = generate_all_bases([n1, n2], [l1])

    vault = tmp_path / "test-vault"
    export_concept_memory_to_obsidian(
        [n1, n2],
        [l1],
        bases,
        vault_path=vault,
    )

    concepts_dir = vault / "Field-State" / "55_Concept_Memory" / "Concepts"
    assert concepts_dir.exists()

    md_files = list(concepts_dir.glob("*.md"))
    assert len(md_files) >= 1


def test_human_block_preservation_on_concept_reexport(tmp_path: Path):
    """Re-exporting preserves custom content added inside the human block."""
    n1 = _node("concept:method:spike-sorting", "spike sorting", "method")
    bases = generate_all_bases([n1], [])

    vault = tmp_path / "test-vault"

    # First export
    export_concept_memory_to_obsidian([n1], [], bases, vault_path=vault)

    # Locate the exported concept note
    concepts_dir = vault / "Field-State" / "55_Concept_Memory" / "Concepts"
    note_files = list(concepts_dir.glob("*.md"))
    assert note_files, "Expected at least one concept note after first export"
    note_path = note_files[0]

    # Inject custom content into the human block
    original_text = note_path.read_text(encoding="utf-8")
    human_annotation = "## My custom note\n\nThis is a human observation.\n"
    modified_text = original_text.replace(
        f"{HUMAN_BEGIN}\n",
        f"{HUMAN_BEGIN}\n\n{human_annotation}",
    )
    note_path.write_text(modified_text, encoding="utf-8")

    # Second export
    export_concept_memory_to_obsidian([n1], [], bases, vault_path=vault)

    # Human content must be preserved
    final_text = note_path.read_text(encoding="utf-8")
    assert "This is a human observation." in final_text
