"""DOI extraction utilities for DANDI, OpenNeuro, and free text sources."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a raw DOI prefix: 10.<registrant>/<suffix>
# The suffix allows any printable non-whitespace chars, stopping before
# trailing punctuation that isn't part of the DOI proper.
_DOI_BODY_RE = re.compile(
    r"10\.\d{4,9}/[^\s\"'<>\[\]{},;()]+",
    re.IGNORECASE,
)

# Trailing punctuation chars we strip off DOI matches from free text
_TRAILING_STRIP = ".,"

_MAX_DOIS = 10


def _clean_doi(raw: str) -> str:
    """Strip trailing punctuation from a raw regex match."""
    return raw.rstrip(_TRAILING_STRIP)


def _bare_doi_from_url(url: str) -> str | None:
    """Extract a bare DOI (10.xxxx/...) from a doi.org URL or doi: prefix."""
    lowered = url.strip()
    # Handle https://doi.org/10.xxx or http://doi.org/10.xxx
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if lowered.lower().startswith(prefix):
            candidate = lowered[len(prefix):]
            m = _DOI_BODY_RE.match(candidate)
            if m:
                return _clean_doi(m.group(0))
    # Handle doi:10.xxx
    if lowered.lower().startswith("doi:"):
        candidate = lowered[4:].lstrip()
        m = _DOI_BODY_RE.match(candidate)
        if m:
            return _clean_doi(m.group(0))
    return None


def extract_dois_from_text(text: str) -> list[str]:
    """Extract DOI strings (bare or URL form) from free text using regex.

    Matches patterns like:
    - 10.1234/something
    - https://doi.org/10.1234/something
    - doi:10.1234/something

    Returns bare DOIs (starting with 10.) deduplicated, max 10.
    """
    if not text:
        return []

    seen: set[str] = set()
    results: list[str] = []

    for m in _DOI_BODY_RE.finditer(text):
        doi = _clean_doi(m.group(0))
        if doi not in seen:
            seen.add(doi)
            results.append(doi)
            if len(results) >= _MAX_DOIS:
                break

    return results


def extract_dois_from_dandi_metadata(metadata: dict) -> list[str]:  # type: ignore[type-arg]
    """Extract DOIs from DANDI dandiset metadata dict.

    DANDI stores related publications in metadata["relatedResource"] as a list
    of dicts:
    [{"relation": "isDescribedBy", "url": "https://doi.org/10.1234/...",
      "schemaKey": "Resource"}, ...]

    Also checks:
    - metadata["doi"] (direct DOI field)
    - metadata["url"] if it contains doi.org
    - metadata["description"] via extract_dois_from_text()

    Returns bare DOIs deduplicated.
    """
    seen: set[str] = set()
    results: list[str] = []

    def _add(doi: str) -> None:
        d = doi.strip()
        if d and d not in seen:
            seen.add(d)
            results.append(d)

    # Direct DOI field
    direct_doi = metadata.get("doi")
    if direct_doi and isinstance(direct_doi, str):
        candidate = _bare_doi_from_url(direct_doi) or (direct_doi if direct_doi.startswith("10.") else None)
        if candidate:
            _add(candidate)

    # relatedResource list
    for resource in metadata.get("relatedResource", []) or []:
        if not isinstance(resource, dict):
            continue
        url = resource.get("url") or resource.get("identifier") or ""
        if isinstance(url, str) and url:
            bare = _bare_doi_from_url(url)
            if bare:
                _add(bare)
            elif url.startswith("10."):
                _add(_clean_doi(url))

    # metadata["url"] if doi.org present
    meta_url = metadata.get("url")
    if meta_url and isinstance(meta_url, str) and "doi.org" in meta_url.lower():
        bare = _bare_doi_from_url(meta_url)
        if bare:
            _add(bare)

    # description fallback
    description = metadata.get("description")
    if description and isinstance(description, str):
        for doi in extract_dois_from_text(description):
            _add(doi)

    return results


def extract_dois_from_openneuro_metadata(metadata: dict) -> list[str]:  # type: ignore[type-arg]
    """Extract DOIs from OpenNeuro dataset metadata.

    OpenNeuro stores publications in:
    - metadata["dataset"]["description"]["ReferencesAndLinks"] list
    - top-level "doi" field
    - metadata["studyLaunchDate"] / metadata["doi"]

    Also checks description text via extract_dois_from_text().
    Returns bare DOIs deduplicated.
    """
    seen: set[str] = set()
    results: list[str] = []

    def _add(doi: str) -> None:
        d = doi.strip()
        if d and d not in seen:
            seen.add(d)
            results.append(d)

    def _try_url(value: str) -> None:
        if not isinstance(value, str) or not value:
            return
        bare = _bare_doi_from_url(value)
        if bare:
            _add(bare)
        elif value.startswith("10."):
            _add(_clean_doi(value))
        elif "doi" in value.lower():
            # Could be embedded DOI in a citation string
            for doi in extract_dois_from_text(value):
                _add(doi)

    # Top-level doi field
    top_doi = metadata.get("doi")
    if top_doi and isinstance(top_doi, str):
        _try_url(top_doi)

    # metadata["dataset"]["description"]["ReferencesAndLinks"]
    dataset_block = metadata.get("dataset")
    if isinstance(dataset_block, dict):
        desc_block = dataset_block.get("description")
        if isinstance(desc_block, dict):
            for ref in desc_block.get("ReferencesAndLinks", []) or []:
                if isinstance(ref, str):
                    _try_url(ref)

    # Flat description.ReferencesAndLinks (alternate layout)
    desc_block2 = metadata.get("description")
    if isinstance(desc_block2, dict):
        for ref in desc_block2.get("ReferencesAndLinks", []) or []:
            if isinstance(ref, str):
                _try_url(ref)
    elif isinstance(desc_block2, str) and desc_block2:
        for doi in extract_dois_from_text(desc_block2):
            _add(doi)

    return results


def dois_to_paper_ids(dois: list[str]) -> list[str]:
    """Convert DOI list to paper node IDs in the format 'paper:doi:{doi_slug}'.

    doi_slug replaces '/' with ':' and lowercases.

    Example:
        '10.1038/s41593-019-0409-2' -> 'paper:doi:10.1038:s41593-019-0409-2'
    """
    return [f"paper:doi:{doi.lower().replace('/', ':')}" for doi in dois]
