"""FastAPI composition root for Neural Search.

`apps.api.main` still contains legacy route implementations. New infrastructure
and domain routers are composed here so the large legacy module can be migrated
incrementally without coupling new services back into it.
"""

from __future__ import annotations

import os

# The legacy app reads NEURAL_SEARCH_DEMO_MODE at import time. Make the public
# execution profile behavioral, not decorative: an unspecified runtime is the
# portable demo, and an explicit demo profile must never opportunistically load
# locally present production corpus files.
_active_profile = os.getenv("NEURAL_SEARCH_PROFILE", "demo").strip().lower()
if _active_profile == "demo":
    os.environ.setdefault("NEURAL_SEARCH_DEMO_MODE", "1")

from apps.api.main import app  # noqa: E402
from apps.api.runtime_router import router as runtime_router  # noqa: E402

app.include_router(runtime_router)

__all__ = ["app"]
