"""NVIDIA NIM provider.

NIM exposes an OpenAI-compatible API, so this provider intentionally stays thin.  The
separate class exists to give Neural Search a stable place for NIM-specific defaults,
health metadata, and future capability negotiation without leaking vendor details into
scientific workflows.
"""

from __future__ import annotations

from neural_search.inference.providers.openai_compatible import OpenAICompatibleProvider
from neural_search.inference.schemas import ProviderSettings


class NIMProvider(OpenAICompatibleProvider):
    """OpenAI-compatible NVIDIA NIM endpoint with explicit provider identity."""

    def __init__(self, settings: ProviderSettings, **kwargs: object):
        if settings.kind not in {"nim", "openai_compatible"}:
            raise ValueError(f"NIM provider requires kind=nim, got {settings.kind}")
        super().__init__(settings, **kwargs)
