"""Shared config loader for literature-linking API clients.

Built now rather than left as ad hoc `os.environ.get(...)` calls per module
(the repo's only prior precedent, `EBRAINS_TOKEN` in
neural_search/ingestion/ebrains.py) because this initiative introduces six
credential/politeness knobs across five external services in one pass --
past that point, ad hoc per-module env lookups drift (three different
literal "mailto" strings already existed in this codebase before this
module existed, from purely organic per-file growth). See .env.example at
the repo root for the documented list of variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureAPIConfig:
    contact_email: str
    crossref_mailto: str
    semantic_scholar_api_key: str | None
    ncbi_api_key: str | None
    ncbi_tool_email: str
    bluesky_handle: str | None
    bluesky_app_password: str | None


def load_literature_api_config() -> LiteratureAPIConfig:
    contact = os.environ.get("NEURAL_SEARCH_CONTACT_EMAIL", "sid.soccer.21@gmail.com")
    return LiteratureAPIConfig(
        contact_email=contact,
        crossref_mailto=os.environ.get("CROSSREF_MAILTO", contact),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        ncbi_api_key=os.environ.get("NCBI_API_KEY") or None,
        ncbi_tool_email=os.environ.get("NCBI_TOOL_EMAIL", contact),
        bluesky_handle=os.environ.get("BLUESKY_HANDLE") or None,
        bluesky_app_password=os.environ.get("BLUESKY_APP_PASSWORD") or None,
    )
