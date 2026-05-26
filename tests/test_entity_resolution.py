from neural_search.normalized import make_dataset_id, make_paper_id
from neural_search.resolve.entities import (
    canonical_entity_key,
    identifier_keys_for_record,
    normalize_doi,
    normalize_openalex_id,
    resolve_entities,
)
from neural_search.schemas import NormalizedDatasetRecord, NormalizedPaperRecord


def test_dataset_identifiers_include_stable_dataset_and_source_keys():
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("DANDI", "000026"),
        source="DANDI",
        source_id="000026",
        title="Mouse OFC reversal learning",
    )

    identifiers = identifier_keys_for_record(record)

    assert identifiers["dataset_id"] == ("dataset:dandi:000026",)
    assert identifiers["source"] == ("dandi:000026",)
    assert canonical_entity_key(record) == "dataset:dandi:000026"


def test_paper_doi_and_openalex_identifiers_are_normalized():
    record = NormalizedPaperRecord(
        paper_id=make_paper_id("openalex", "W123"),
        source="openalex",
        source_id="https://openalex.org/W123",
        title="Decision making paper",
        doi="https://doi.org/10.1101/ABC.DEF.",
    )

    identifiers = identifier_keys_for_record(record)

    assert normalize_doi(record.doi) == "10.1101/abc.def"
    assert normalize_openalex_id(record.source_id) == "W123"
    assert identifiers["doi"] == ("10.1101/abc.def",)
    assert identifiers["openalex"] == ("W123",)
    assert canonical_entity_key(record) == "paper:doi:10.1101/abc.def"


def test_resolve_entities_merges_duplicate_datasets_by_source_identifier():
    left = {
        "dataset_id": "dataset:dandi:000026",
        "source": "dandi",
        "source_id": "000026",
        "title": "Mouse OFC reversal learning",
    }
    right = {
        "dataset_id": "dataset:dandi:DANDI_000026_ALIAS",
        "source": "DANDI",
        "source_id": "000026",
        "title": "Mouse OFC reversal learning curated alias",
    }

    report = resolve_entities([left, right])

    assert len(report.entities) == 1
    entity = report.entities[0]
    assert entity.canonical_id == "dataset:dandi:000026"
    assert entity.record_ids == (
        "dataset:dandi:000026",
        "dataset:dandi:DANDI_000026_ALIAS",
    )
    assert report.duplicate_groups == (entity.record_ids,)
    assert report.lookup["source:dandi:000026"] == "dataset:dandi:000026"
    assert entity.conflicts[0].field == "title"


def test_resolve_entities_merges_papers_by_doi_across_openalex_records():
    left = {
        "paper_id": "paper:openalex:W123",
        "source": "openalex",
        "source_id": "W123",
        "title": "Neural dynamics paper",
        "doi": "10.1038/example",
    }
    right = {
        "paper_id": "paper:crossref:10.1038_example",
        "source": "crossref",
        "source_id": "10.1038/example",
        "title": "Neural dynamics paper",
        "doi": "https://doi.org/10.1038/EXAMPLE",
    }

    report = resolve_entities([left, right])

    assert len(report.entities) == 1
    entity = report.entities[0]
    assert entity.entity_type == "paper"
    assert entity.canonical_id == "paper:doi:10.1038/example"
    assert entity.identifiers["openalex"] == ("W123",)
    assert report.lookup["doi:10.1038/example"] == "paper:doi:10.1038/example"
    assert entity.conflicts[0].field == "source"


def test_resolve_entities_keeps_unlinked_records_separate_and_stable():
    records = [
        {
            "dataset_id": "dataset:dandi:000001",
            "source": "dandi",
            "source_id": "000001",
            "title": "Mouse Neuropixels",
        },
        {
            "paper_id": "paper:openalex:W999",
            "source": "openalex",
            "source_id": "W999",
            "title": "Standalone paper",
        },
    ]

    report = resolve_entities(records)

    assert [entity.canonical_id for entity in report.entities] == [
        "dataset:dandi:000001",
        "paper:openalex:W999",
    ]
    assert report.duplicate_groups == ()
    assert report.conflicts == ()


def test_resolve_entities_empty_input_is_deterministic():
    report = resolve_entities([])

    assert report.entities == ()
    assert report.lookup == {}
    assert report.duplicate_groups == ()
