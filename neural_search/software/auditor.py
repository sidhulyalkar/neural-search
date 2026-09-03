"""Model-assisted code hypothesis generation with strict provenance boundaries."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from neural_search.inference import (
    InferenceCapability,
    InferenceMessage,
    InferenceRequest,
    InferenceService,
)
from neural_search.software.schema import AuditHypothesis, AuditState, SoftwareComponent


class CodeAuditInput(BaseModel):
    component: SoftwareComponent
    source_text: str
    repository_revision: str
    algorithm_context: str | None = None
    literature_context: list[str] = Field(default_factory=list)
    known_tests: list[str] = Field(default_factory=list)


AUDIT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "rationale", "confidence", "testable_prediction"],
    "properties": {
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "testable_prediction": {"type": "string"},
        "affected_conditions": {"type": "array", "items": {"type": "string"}},
        "candidate_oracle": {"type": ["string", "null"]},
    },
}


def build_code_audit_messages(item: CodeAuditInput) -> list[InferenceMessage]:
    """Build a skeptical audit prompt that asks for testable hypotheses, not verdicts."""

    system = (
        "You are reviewing scientific software for testable numerical or algorithmic "
        "hypotheses. Do not declare a bug, affected paper, or scientific impact. Identify "
        "at most one high-value behavior that can be independently verified. Prefer no "
        "hypothesis over a speculative one. Return only the requested structured object."
    )
    context = {
        "component": item.component.model_dump(mode="json"),
        "repository_revision": item.repository_revision,
        "algorithm_context": item.algorithm_context,
        "literature_context": item.literature_context,
        "known_tests": item.known_tests,
        "source_text": item.source_text,
    }
    return [
        InferenceMessage(role="system", content=system),
        InferenceMessage(role="user", content=json.dumps(context, indent=2, sort_keys=True)),
    ]


def generate_audit_hypothesis(
    service: InferenceService,
    item: CodeAuditInput,
    *,
    model_profile: str | None = None,
) -> AuditHypothesis:
    """Generate one unverified hypothesis and preserve the model invocation as provenance."""

    request = InferenceRequest(
        messages=build_code_audit_messages(item),
        capability=InferenceCapability.CODE_REASONING,
        model_profile=model_profile,
        response_schema=AUDIT_RESPONSE_SCHEMA,
        metadata={
            "input_revision": item.repository_revision,
            "prompt_template": "scientific_code_audit_v1",
            "component_id": item.component.component_id,
        },
    )
    result = service.generate(request)
    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ValueError("code audit model did not return valid JSON") from exc
    summary = str(payload.get("summary", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    if not summary or not rationale:
        raise ValueError("code audit model returned an empty hypothesis")
    confidence = float(payload.get("confidence", 0.0))
    return AuditHypothesis(
        hypothesis_id=f"hypothesis:{item.component.component_id}:{result.manifest.run_id}",
        component_id=item.component.component_id,
        summary=summary,
        rationale=rationale,
        state=AuditState.CODE_REVIEW_HYPOTHESIS,
        source_run_ids=[result.manifest.run_id],
        metadata={
            "confidence": max(0.0, min(1.0, confidence)),
            "testable_prediction": payload.get("testable_prediction"),
            "affected_conditions": payload.get("affected_conditions", []),
            "candidate_oracle": payload.get("candidate_oracle"),
            "repository_revision": item.repository_revision,
            "inference_manifest": result.manifest.model_dump(mode="json"),
        },
    )
