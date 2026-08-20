"""FastAPI composition root for Neural Search.

`apps.api.main` still contains legacy route implementations. New infrastructure
and domain routers are composed here so the large legacy module can be migrated
incrementally without coupling new services back into it.

Execution profiles are capability contracts, not hidden configuration switches.
Data-source selection remains explicit in the legacy application until it is
migrated behind an app factory/settings boundary.
"""

from __future__ import annotations

from apps.api.main import app
from apps.api.runtime_router import router as runtime_router

app.include_router(runtime_router)

__all__ = ["app"]
