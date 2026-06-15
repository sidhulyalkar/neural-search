"""LLM-based query expansion fallback.

When rule-based ontology parsing produces zero brain_regions AND zero tasks,
this module calls Claude claude-haiku-4-5 to extract structured neuro terms from
the query text. The result is merged back into parsed_query before scoring.

Configuration (in retrieval.yaml):
    llm_expansion:
      enabled: false      # opt-in only
      model: claude-haiku-4-5-20251001
      max_tokens: 256
      cache_size: 512     # LRU cache entries

Environment:
    ANTHROPIC_API_KEY    Required when enabled=true

Provenance: all LLM-inferred terms carry dimension key prefix "llm_inferred:"
  so they can be distinguished from rule-based matches in score_breakdown.

Scientific accuracy constraints (must NOT produce):
  - calcium_imaging = spike_sorting (different modalities)
  - fmri = lfp (different scales)
  - entorhinal_cortex = hippocampus (different regions)
  - human = rodent (different species)
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a neuroscience dataset expert. Extract structured metadata terms from a "
    "search query. Return ONLY valid JSON with these keys: brain_regions (canonical "
    "snake_case IDs like hippocampus, visual_cortex), tasks (behavioral task IDs), "
    "modalities (neuropixels, fmri, calcium_imaging, etc.), species (mouse, rat, human, "
    "macaque, etc.). Use empty lists when terms are absent. Do NOT invent terms."
)

_VALID_KEYS = frozenset({"brain_regions", "tasks", "modalities", "species"})


def expand_query_with_llm(
    query: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 256,
) -> dict[str, list[str]]:
    """Return {brain_regions, tasks, modalities, species} extracted by LLM.

    Returns empty dict on any error so search continues without expansion.
    """
    return _cached_expand(query, model=model, max_tokens=max_tokens)


@lru_cache(maxsize=512)
def _cached_expand(
    query: str,
    *,
    model: str,
    max_tokens: int,
) -> dict[str, list[str]]:
    """LRU-cached inner implementation keyed on (query, model, max_tokens)."""
    try:
        anthropic = _import_anthropic()
    except ImportError:
        logger.warning(
            "anthropic package not installed; LLM query expansion is unavailable"
        )
        return {}

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Query: {query}\nReturn JSON only."}
            ],
        )
        raw_text: str = message.content[0].text
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM expansion API call failed: %s", exc)
        return {}

    parsed = _parse_json_response(raw_text, query)
    if not parsed:
        return {}

    result = _validate_and_filter(parsed)
    logger.debug(
        "LLM expansion for query %r produced: %s",
        query,
        dict(result),
    )
    return result


def _import_anthropic():
    """Lazy import of the anthropic package."""
    import anthropic  # noqa: PLC0415

    return anthropic


def _parse_json_response(text: str, query: str) -> dict | None:
    """Attempt to parse JSON from an LLM response, stripping markdown fences."""
    # First try the raw text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Last resort: extract the first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(
        "LLM expansion could not parse JSON response for query %r: %.120s",
        query,
        text,
    )
    return None


def _validate_and_filter(raw: dict) -> dict[str, list[str]]:
    """Return only recognised keys whose values are non-empty lists of strings."""
    result: dict[str, list[str]] = {}
    for key in _VALID_KEYS:
        value = raw.get(key)
        if not isinstance(value, list):
            continue
        # Keep only string elements; silently drop integers, None, etc.
        strings = [item for item in value if isinstance(item, str)]
        if strings:
            result[key] = strings
    return result
