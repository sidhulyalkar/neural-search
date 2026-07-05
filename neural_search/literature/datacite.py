"""Link corpus datasets to papers via DataCite's structured, authoritative
`relatedIdentifiers` -- no fuzzy title matching needed.

DataCite is the DOI registration agency for most repository-issued dataset
DOIs (Zenodo, Figshare, OSF, Harvard Dataverse, DANDI). A dataset's DataCite
record can directly declare an `IsCitedBy`/`IsSupplementTo`/`IsDescribedBy`
relation to a paper DOI -- confirmed via a live lookup during development
(2026-07-02): `10.5281/zenodo.11236154` declares
`IsSupplementTo -> 10.1038/s41467-024-49226-9` (a real Nature Communications
paper), with `resourceTypeGeneral: "JournalArticle"` on that relation.

Only datasets with a `doi` field in the normalized corpus record are
queryable this way. Confirmed directly against the real corpus
(2026-07-02): DANDI and OpenNeuro records do not carry a `doi` field at all
(their DataCite-registered DOIs, e.g. DANDI's `10.48324/dandi.*` pattern,
are not captured by corpus normalization today) -- only neuromorpho, zenodo,
figshare, osf, and harvard_dataverse records have one (2,387/7,171). This is
a real, honest gap in corpus normalization, not a DataCite API limitation;
datasets without a `doi` field get `match_method="not_applicable_no_dataset_doi"`
so they are distinguishable from a genuine DataCite lookup miss
(`match_method="not_found"`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from neural_search.literature.api_client import (
    TransientLookupError,
    get_or_raise_transient,
)
from neural_search.literature.corpus_io import (
    DatasetPaperLink,
    iter_corpus_records,
    make_not_found_link,
)

_DATACITE_BASE = "https://api.datacite.org/dois"
_RELEVANT_RELATION_TYPES = {"IsCitedBy", "IsSupplementTo", "IsDescribedBy"}


def fetch_datacite_record(doi: str) -> dict | None:
    """GET a DataCite DOI record. Returns None only on a genuine 404.

    Raises TransientLookupError on 429/5xx -- callers must not treat that as
    "not found" (same discipline as neural_search.literature.linking's
    OpenAlex client, after the incident documented there).
    """

    resp = get_or_raise_transient(f"{_DATACITE_BASE}/{doi}", source="DataCite")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def extract_related_paper_dois(record: dict) -> list[dict]:
    """Return declared paper-relation DOIs from a DataCite record.

    Filters `relatedIdentifiers` to relation types that indicate the related
    item is a paper describing/citing/supplementing this dataset, and to
    DOI-typed identifiers only (DataCite also allows URL/ISBN/etc. relation
    targets, which this plan does not resolve).
    """

    attributes = record.get("data", {}).get("attributes", {})
    related = attributes.get("relatedIdentifiers") or []
    hits: list[dict] = []
    for rel in related:
        if rel.get("relationType") not in _RELEVANT_RELATION_TYPES:
            continue
        if rel.get("relatedIdentifierType") != "DOI":
            continue
        doi = rel.get("relatedIdentifier")
        if not doi:
            continue
        hits.append(
            {
                "doi": doi,
                "relation_type": rel["relationType"],
                "resource_type_general": rel.get("resourceTypeGeneral"),
            }
        )
    return hits


def link_corpus_to_datacite(corpus_path: Path, out_path: Path) -> list[DatasetPaperLink]:
    """Link corpus datasets to papers via DataCite relatedIdentifiers.

    Stops immediately on TransientLookupError, matching
    link_corpus_to_literature's behavior -- records resolved before the
    failure are still written to `out_path`.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    links: list[DatasetPaperLink] = []

    with out_path.open("w") as out_fh:
        for i, rec in enumerate(iter_corpus_records(corpus_path)):
            record_id = f"{rec['source']}:{rec['source_id']}"
            dataset_doi = rec.get("doi") or None

            if not dataset_doi:
                # Distinct match_method (not the generic "not_found") so this
                # is distinguishable from a genuine DataCite lookup miss --
                # the dataset was never queryable via this source at all.
                link = make_not_found_link(record_id, paper_source="datacite")
                link.match_method = "not_applicable_no_dataset_doi"
            else:
                try:
                    link = _resolve_datacite_link(record_id, dataset_doi)
                except TransientLookupError as exc:
                    print(
                        f"Aborting after {i} records: transient DataCite lookup "
                        f"failure ({exc}). Records processed so far are saved; "
                        f"re-run once resolved rather than trusting the "
                        f"remaining corpus as genuinely unmatched."
                    )
                    break

            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)

    return links


def _resolve_datacite_link(record_id: str, dataset_doi: str) -> DatasetPaperLink:
    record = fetch_datacite_record(dataset_doi)
    if record is None:
        return make_not_found_link(record_id, paper_source="datacite")

    related = extract_related_paper_dois(record)
    if not related:
        return make_not_found_link(record_id, paper_source="datacite")

    # A dataset can declare multiple related-paper relations; take the first
    # -- reanalysis_bridge_builder.py's precedent-matching and
    # paper_node_builder.py both operate at the (dataset, paper) granularity
    # already used by every other source in this plan. If a dataset needs
    # multiple DataCite-declared papers in a future phase, this is the single
    # place to extend to return multiple links per dataset.
    hit = related[0]
    return DatasetPaperLink(
        dataset_record_id=record_id,
        paper_openalex_id="",
        paper_doi=hit["doi"],
        paper_title=None,
        paper_year=None,
        match_method="datacite_related_identifier",
        confidence=1.0,
        paper_source="datacite",
        paper_source_id=hit["doi"],
    )
