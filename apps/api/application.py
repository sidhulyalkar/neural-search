"""FastAPI composition root for Neural Search.

`apps.api.main` still contains legacy route implementations. New infrastructure
and domain routers should be composed here so the large legacy module can be
migrated incrementally without coupling new services back into it.
"""

from __future__ import annotations

from apps.api.main import app
from apps.api.runtime_router import router as runtime_router

# New routers belong at the composition boundary rather than inside the legacy
# route module. FastAPI rejects neither duplicate tags nor late router inclusion,
# so this remains backward-compatible with callers that still import main:app.
app.include_router(runtime_router)

__all__ = ["app"]
