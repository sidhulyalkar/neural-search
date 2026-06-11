"""Optional LLM judge for silver qrels generation.

Disabled by default.  Requires --use-llm-judge flag and ANTHROPIC_API_KEY.
Tests use the MockLLMJudge; the real judge is skipped when the API key is absent.

The judge sees ONLY:
  - query text and scientific goal
  - must-have constraints
  - hard-negative descriptions
  - dataset metadata (title, description, modalities, species, tasks, etc.)
  - concept explanation (if available)
  - the relevance rubric (0–3 scale)

The judge must NOT see:
  - retrieval score or rank
  - variant name (bm25, usefulness, concept_full, etc.)
  - whether the concept reranker surfaced the result
  - any existing qrel labels
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from scripts.eval.silver_qrels_schema import LabelingFunctionVote

# ---------------------------------------------------------------------------
# Relevance rubric (shown to judge in every prompt)
# ---------------------------------------------------------------------------

RELEVANCE_RUBRIC = """
Relevance scale for neuroscience dataset search:

0 — Not relevant. The dataset does not support the stated scientific goal.
    May match on surface keywords but fails on species, modality, or task.
1 — Marginally relevant. Could be loosely related but has significant gaps
    in species, modality, task, or analytical affordance.
2 — Relevant. Supports the scientific goal with minor caveats or missing
    metadata. A researcher would likely consider this dataset.
3 — Highly relevant. Strongly matches species, modality, task, and
    analytical affordances. A researcher would almost certainly use this.

Rules:
- A hard-negative violation (e.g. wrong species, incompatible modality)
  MUST be scored 0 regardless of other matches.
- Absence of metadata should lower your confidence, not automatically lower
  your relevance score.
"""

JUDGE_PROMPT_TEMPLATE = """You are a neuroscience dataset relevance assessor.

## Search query
{query_text}

## Scientific goal
{scientific_goal}

## Must-have constraints
{must_have}

## Hard-negative patterns (these patterns indicate NOT relevant → score 0)
{hard_negatives}

## Dataset metadata
Title: {title}
Source: {source} / {source_id}
Description: {description}
Species: {species}
Modalities: {modalities}
Brain regions: {brain_regions}
Tasks: {tasks}
Data standards: {data_standards}
License: {license}

{concept_section}

## Relevance rubric
{rubric}

## Your task
Return a JSON object with these fields:
- relevance: integer 0–3
- confidence: float 0.0–1.0
- rationale: one or two sentences explaining your score
- hard_negative_violation: true/false
- missing_metadata: list of missing fields that limit your confidence
- evidence_fields_used: list of dataset fields you used in your decision

Respond ONLY with valid JSON. No other text.
"""


# ---------------------------------------------------------------------------
# Protocol and mock
# ---------------------------------------------------------------------------


class LLMJudgeProtocol(Protocol):
    def judge(
        self,
        query: Any,
        record: dict[str, Any],
        concept_result: dict[str, Any] | None = None,
    ) -> LabelingFunctionVote: ...


class MockLLMJudge:
    """Deterministic mock judge for testing.  Returns fixed votes based on
    record modality to exercise the pipeline without API calls."""

    def judge(
        self,
        query: Any,
        record: dict[str, Any],
        concept_result: dict[str, Any] | None = None,
    ) -> LabelingFunctionVote:
        # Simple deterministic logic: match first modality keyword
        mods = [str(m).lower() for m in (record.get("modalities") or [])]
        expected = [str(e).lower() for e in getattr(query, "expected_modalities", [])]

        if expected and any(any(e in m for e in expected) for m in mods):
            vote, confidence = 2, 0.70
            rationale = "mock judge: modality match detected"
        elif mods:
            vote, confidence = 1, 0.50
            rationale = "mock judge: modality present but may not match"
        else:
            vote, confidence = None, 0.30
            rationale = "mock judge: insufficient metadata to judge"

        return LabelingFunctionVote(
            source="llm_judge",
            vote=vote,
            confidence=confidence,
            rationale=rationale,
            evidence=[f"mock_judge_mod: {mods}"],
        )


class AnthropicLLMJudge:
    """Real LLM judge using the Anthropic API.

    Requires ANTHROPIC_API_KEY environment variable.
    Skips gracefully if anthropic package is not installed.
    """

    MODEL = "claude-haiku-4-5-20251001"  # use lightweight model for judging

    def __init__(self) -> None:
        self._client: Any = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return
        try:
            import anthropic  # noqa: T201  # optional dep; not in requirements
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def judge(
        self,
        query: Any,
        record: dict[str, Any],
        concept_result: dict[str, Any] | None = None,
    ) -> LabelingFunctionVote:
        if not self._available:
            return LabelingFunctionVote(
                source="llm_judge",
                vote=None,
                confidence=0.0,
                rationale="llm_judge unavailable (no API key or anthropic package)",
            )

        concept_section = ""
        if concept_result and concept_result.get("explanation_summary"):
            concept_section = (
                f"## Concept memory explanation\n{concept_result['explanation_summary']}"
            )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query_text=getattr(query, "query_text", ""),
            scientific_goal=getattr(query, "scientific_goal", ""),
            must_have="\n".join(f"- {m}" for m in getattr(query, "must_have", [])),
            hard_negatives="\n".join(f"- {h}" for h in getattr(query, "hard_negatives", [])),
            title=record.get("title", ""),
            source=record.get("source", ""),
            source_id=record.get("source_id", ""),
            description=(str(record.get("description", "") or "")[:400]),
            species=", ".join(str(s) for s in (record.get("species") or [])),
            modalities=", ".join(str(m) for m in (record.get("modalities") or [])),
            brain_regions=", ".join(str(r) for r in (record.get("brain_regions") or [])),
            tasks=", ".join(str(t) for t in (record.get("tasks") or [])),
            data_standards=", ".join(str(s) for s in (record.get("data_standards") or [])),
            license=record.get("license", ""),
            concept_section=concept_section,
            rubric=RELEVANCE_RUBRIC,
        )

        try:
            import json as _json
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            data = _json.loads(text)

            return LabelingFunctionVote(
                source="llm_judge",
                vote=int(data.get("relevance", None)) if data.get("relevance") is not None else None,
                confidence=float(data.get("confidence", 0.5)),
                rationale=str(data.get("rationale", "")),
                evidence=(
                    [f"hn_violation: {data['hard_negative_violation']}"]
                    + [f"missing: {m}" for m in (data.get("missing_metadata") or [])]
                    + [f"used: {f}" for f in (data.get("evidence_fields_used") or [])]
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return LabelingFunctionVote(
                source="llm_judge",
                vote=None,
                confidence=0.0,
                rationale=f"llm_judge error: {exc}",
            )


def build_judge(use_llm_judge: bool = False) -> LLMJudgeProtocol | None:
    """Return the appropriate judge or None if LLM judging is disabled."""
    if not use_llm_judge:
        return None
    judge = AnthropicLLMJudge()
    if not judge.available:
        return None
    return judge
