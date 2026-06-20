"""Validate that the qrels pipeline key discovery works.

Note: the module caches env vars in module-level globals at import time and also
calls load_dotenv(.env.local) on reload, so importlib.reload cannot cleanly
suppress .env.local values.  Instead we patch the module globals directly after
import to control the discovery logic without touching the filesystem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import scripts.eval.run_parallel_llm_qrels as _qrels_mod  # noqa: E402


def test_discover_workers_uses_openrouter_key_from_env():
    """_discover_workers should return OpenRouter workers when OPENROUTER_API_KEY is set."""
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-testkey"}),
        patch.object(_qrels_mod, "_OLLAMA_BASE_URL", ""),
        patch.object(_qrels_mod, "_LM_STUDIO_BASE_URL", ""),
        patch.object(_qrels_mod, "_KEYWAY_BASE_URL", ""),
    ):
        workers = _qrels_mod._discover_workers()
    assert len(workers) > 0, "Should have at least one worker"
    assert all(w.api_key == "sk-or-v1-testkey" for w in workers)


def test_discover_workers_returns_empty_when_no_key():
    """_discover_workers should return [] when no keys are configured."""
    # Build a clean env that strips all known key env vars
    clean_env: dict[str, str] = {"OPENROUTER_API_KEY": ""}
    for k in list(os.environ.keys()):
        if any(k.startswith(p) for p in ["CLAUDE_OPUS", "GEMINI_API_KEY"]):
            clean_env[k] = ""
    with (
        patch.dict(os.environ, clean_env),
        patch.object(_qrels_mod, "_OLLAMA_BASE_URL", ""),
        patch.object(_qrels_mod, "_LM_STUDIO_BASE_URL", ""),
        patch.object(_qrels_mod, "_KEYWAY_BASE_URL", ""),
    ):
        workers = _qrels_mod._discover_workers()
    assert workers == [], f"Expected empty list, got {workers}"
