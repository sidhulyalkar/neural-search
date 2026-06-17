"""Tests for the finding extraction CLI helpers."""

from __future__ import annotations

import os

from scripts.literature.extract_findings import _resolve_anthropic_api_key


def test_resolve_api_key_uses_claude_opus_fallback(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.local"
    env_path.write_text("CLAUDE_OPUS_API_KEY=test-opus-key\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_OPUS_API_KEY", raising=False)

    from scripts.literature import extract_findings

    original_loader = extract_findings._load_env_local
    monkeypatch.setattr(extract_findings, "_load_env_local", lambda: original_loader(env_path))

    assert _resolve_anthropic_api_key() == "test-opus-key"
    assert os.environ["ANTHROPIC_API_KEY"] == "test-opus-key"
