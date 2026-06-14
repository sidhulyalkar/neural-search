"""Tests for LLM-based query expansion fallback."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_stub(response_text: str) -> types.ModuleType:
    """Build a minimal anthropic stub that returns *response_text* from messages.create."""
    stub = types.ModuleType("anthropic")

    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client_instance = MagicMock()
    client_instance.messages.create.return_value = message

    stub.Anthropic = MagicMock(return_value=client_instance)
    return stub


def _make_error_anthropic_stub(exc: Exception) -> types.ModuleType:
    """Build a stub whose messages.create raises *exc*."""
    stub = types.ModuleType("anthropic")

    client_instance = MagicMock()
    client_instance.messages.create.side_effect = exc

    stub.Anthropic = MagicMock(return_value=client_instance)
    return stub


# ---------------------------------------------------------------------------
# Fixtures — ensure the LRU cache is cleared between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear the LRU cache on the private helper before every test."""
    from neural_search.search import llm_expansion  # noqa: PLC0415

    llm_expansion._cached_expand.cache_clear()
    yield
    llm_expansion._cached_expand.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExpandQueryWithLlm:
    """Tests for expand_query_with_llm and its helpers."""

    # ------------------------------------------------------------------
    # 1. ImportError — anthropic not installed
    # ------------------------------------------------------------------

    def test_returns_empty_dict_when_anthropic_not_installed(self):
        """Returns {} gracefully when the anthropic package cannot be imported."""
        with patch.dict(sys.modules, {"anthropic": None}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("hippocampus theta oscillations")

        assert result == {}

    # ------------------------------------------------------------------
    # 2. API error
    # ------------------------------------------------------------------

    def test_returns_empty_dict_on_api_error(self):
        """Returns {} when the API call raises an exception."""
        stub = _make_error_anthropic_stub(RuntimeError("connection refused"))

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("mouse visual cortex calcium imaging")

        assert result == {}

    # ------------------------------------------------------------------
    # 3. Clean JSON response
    # ------------------------------------------------------------------

    def test_parses_clean_json_response(self):
        """Parses a plain JSON response and returns the extracted terms."""
        payload = {"brain_regions": ["hippocampus"], "tasks": [], "modalities": [], "species": []}
        stub = _make_anthropic_stub(json.dumps(payload))

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("hippocampus place cells")

        assert result.get("brain_regions") == ["hippocampus"]
        # Empty lists should be dropped
        assert "tasks" not in result
        assert "modalities" not in result
        assert "species" not in result

    # ------------------------------------------------------------------
    # 4. Markdown-wrapped JSON
    # ------------------------------------------------------------------

    def test_handles_markdown_wrapped_json(self):
        """Strips markdown code fences before parsing JSON."""
        payload = {"brain_regions": ["visual_cortex"], "tasks": ["passive_viewing"]}
        markdown_response = f"```json\n{json.dumps(payload)}\n```"
        stub = _make_anthropic_stub(markdown_response)

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("visual cortex passive viewing mouse")

        assert "visual_cortex" in result.get("brain_regions", [])
        assert "passive_viewing" in result.get("tasks", [])

    # ------------------------------------------------------------------
    # 5. Non-string values are filtered out
    # ------------------------------------------------------------------

    def test_filters_non_string_values(self):
        """Non-string items inside a list are silently dropped."""
        payload = {"brain_regions": [1, "hippocampus", None, True, "prefrontal_cortex"]}
        stub = _make_anthropic_stub(json.dumps(payload))

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("prefrontal cortex working memory")

        brain_regions = result.get("brain_regions", [])
        assert "hippocampus" in brain_regions
        assert "prefrontal_cortex" in brain_regions
        # Non-strings must be absent
        for item in brain_regions:
            assert isinstance(item, str)

    # ------------------------------------------------------------------
    # 6. Empty lists are dropped from the result
    # ------------------------------------------------------------------

    def test_returns_only_nonempty_keys(self):
        """Keys with empty lists are omitted from the returned dict."""
        payload = {
            "brain_regions": ["hippocampus"],
            "tasks": [],
            "modalities": [],
            "species": ["mouse"],
        }
        stub = _make_anthropic_stub(json.dumps(payload))

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("mouse hippocampus place cells sharp wave ripples")

        assert set(result.keys()) == {"brain_regions", "species"}

    # ------------------------------------------------------------------
    # 7. Invalid JSON → empty dict
    # ------------------------------------------------------------------

    def test_invalid_json_returns_empty(self):
        """Malformed LLM response that cannot be parsed returns {}."""
        stub = _make_anthropic_stub("Sorry, I cannot help with that.")

        with patch.dict(sys.modules, {"anthropic": stub}):
            from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

            result = expand_query_with_llm("fmri resting state default mode network")

        assert result == {}

    # ------------------------------------------------------------------
    # 8. Module imports cleanly without anthropic installed
    # ------------------------------------------------------------------

    def test_function_is_callable_without_anthropic(self):
        """The module can be imported and the function called even if anthropic is absent."""
        # The module must already be importable (it has been imported above).
        from neural_search.search.llm_expansion import expand_query_with_llm  # noqa: PLC0415

        assert callable(expand_query_with_llm)

        # Calling it with anthropic absent should return {} without raising
        with patch.dict(sys.modules, {"anthropic": None}):
            result = expand_query_with_llm("entorhinal cortex grid cells")

        assert isinstance(result, dict)
