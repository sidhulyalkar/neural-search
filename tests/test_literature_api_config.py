"""Tests for neural_search.literature.api_config.load_literature_api_config."""

from __future__ import annotations

import pytest

from neural_search.literature.api_config import load_literature_api_config


def test_defaults_when_no_env_vars_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "NEURAL_SEARCH_CONTACT_EMAIL",
        "CROSSREF_MAILTO",
        "SEMANTIC_SCHOLAR_API_KEY",
        "NCBI_API_KEY",
        "NCBI_TOOL_EMAIL",
        "BLUESKY_HANDLE",
        "BLUESKY_APP_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)

    config = load_literature_api_config()

    assert config.contact_email == "sid.soccer.21@gmail.com"
    assert config.crossref_mailto == config.contact_email
    assert config.ncbi_tool_email == config.contact_email
    assert config.semantic_scholar_api_key is None
    assert config.ncbi_api_key is None
    assert config.bluesky_handle is None
    assert config.bluesky_app_password is None


def test_explicit_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEURAL_SEARCH_CONTACT_EMAIL", "team@example.org")
    monkeypatch.setenv("CROSSREF_MAILTO", "crossref@example.org")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-key")
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-key")

    config = load_literature_api_config()

    assert config.contact_email == "team@example.org"
    assert config.crossref_mailto == "crossref@example.org"
    assert config.semantic_scholar_api_key == "s2-key"
    assert config.ncbi_api_key == "ncbi-key"
    # NCBI_TOOL_EMAIL wasn't set explicitly, so it falls back to the contact.
    assert config.ncbi_tool_email == "team@example.org"


def test_empty_string_env_var_treated_as_unset_for_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")

    config = load_literature_api_config()

    assert config.semantic_scholar_api_key is None
