"""Shared corpus-record iteration and link-result types for literature-linking
sources (OpenAlex, DataCite, Crossref, Semantic Scholar, PubMed).

`DatasetPaperLink` lives here (not in linking.py, its original home) so that
this module has no dependency on linking.py -- avoiding a circular import,
since linking.py itself imports the iteration helpers below.
`neural_search.literature.linking` re-imports `DatasetPaperLink` for
backward compatibility; existing callers importing it from `linking` are
unaffected.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetPaperLink:
    dataset_record_id: str
    paper_openalex_id: str
    paper_doi: str | None
    paper_title: str | None
    paper_year: int | None
    match_method: str
    # "doi_exact" | "title_fuzzy" | "title_fuzzy_local" |
    # "datacite_related_identifier" | "crossref_doi_exact" |
    # "semantic_scholar_doi_exact" | "pubmed_doi_exact" | "not_found"
    confidence: float  # 1.0 doi_exact/datacite, 0.7-0.9 title_fuzzy, 0.0 not_found
    paper_source: str = "openalex"
    # "openalex" | "datacite" | "crossref" | "semantic_scholar" | "pubmed" | "biorxiv"
    paper_source_id: str = ""  # source-native ID: OpenAlex W-id, DOI, S2 paperId, PMID

    def __post_init__(self) -> None:
        # Backward compatibility: rows constructed by pre-existing OpenAlex
        # call sites only ever set paper_openalex_id, not paper_source_id --
        # this keeps them addressable by the new paper_source/paper_source_id
        # dispatch without requiring every existing caller to change.
        if not self.paper_source_id and self.paper_openalex_id:
            self.paper_source_id = self.paper_openalex_id


def make_not_found_link(record_id: str, *, paper_source: str = "openalex") -> DatasetPaperLink:
    return DatasetPaperLink(
        dataset_record_id=record_id,
        paper_openalex_id="",
        paper_doi=None,
        paper_title=None,
        paper_year=None,
        match_method="not_found",
        confidence=0.0,
        paper_source=paper_source,
    )


def iter_corpus_records(corpus_path: Path) -> Iterator[dict]:
    paths = (
        sorted(corpus_path.glob("*.jsonl"))
        if corpus_path.is_dir()
        else [corpus_path]
    )
    for p in paths:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_paper_records(paper_shards: Path) -> Iterator[dict]:
    paths = (
        sorted(paper_shards.glob("*.jsonl"))
        if paper_shards.is_dir()
        else [paper_shards]
    )
    for path in paths:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
