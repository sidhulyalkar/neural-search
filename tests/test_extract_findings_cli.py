"""Tests for the finding extraction CLI helpers."""

from __future__ import annotations

import os

from scripts.literature.extract_findings import (
    _resolve_anthropic_api_key,
    discover_providers,
)


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


def test_discover_providers_includes_openrouter_and_deepseek(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=or-key",
                "OPENROUTER_BASE_URL=https://openrouter.ai",
                "DEEPSEEK_API_KEY=ds-key",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for key in [
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "CLAUDE_OPUS_API_KEY",
        "ANTHROPIC_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    from scripts.literature import extract_findings

    original_loader = extract_findings._load_env_local
    monkeypatch.setattr(extract_findings, "_load_env_local", lambda: original_loader(env_path))

    providers = discover_providers()

    by_name = {p.name: p for p in providers}
    assert by_name["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert by_name["deepseek_api_key"].model == "deepseek-chat"
