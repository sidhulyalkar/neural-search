# Task 06 — LLM Rubric Judge (Optional)

**Files:**
- Create: `neural_search/eval/llm_judge.py`
- Create: `configs/judges/rubric_judge_v1.yaml`
- Create: `scripts/eval/judge_candidates.py`

The judge is entirely optional. If `ANTHROPIC_API_KEY` is absent or the
config file is missing, `judge_candidates.py` exits cleanly with a warning.

---

- [ ] **Step 1: Create `configs/judges/rubric_judge_v1.yaml`**

```yaml
# Rubric Judge v1 — configuration for LLM-based relevance judging
version: "1.0"

# Which provider / model to use.
# Requires the corresponding SDK to be installed.
provider: anthropic
model: claude-haiku-4-5-20251001    # cheapest capable model

# Self-consistency: run each pair N times and take majority label.
# 1 = no self-consistency (faster, cheaper).
n_runs: 1

# Weight applied to LLM votes in the ensemble (vs LF votes at 1.0×).
llm_weight: 1.5

# Max pairs to judge (cost guard). Set to null for no limit.
max_pairs: 500

# Rubric text injected verbatim into the system prompt.
rubric: |
  You are a neuroscience dataset relevance judge. Given a research query and a
  dataset description, assign a relevance label on a 0–3 scale:

    3 — Highly relevant: dataset directly supports the query intent.
        All required modalities, species, and data levels are present.
    2 — Partially relevant: dataset partially supports the query.
        Some required properties are present but not all.
    1 — Weakly relevant: dataset is tangentially related but would
        require substantial pre-processing or adaptation.
    0 — Not relevant: dataset does not match the query constraints.
        Hard negative: superficially similar but actually wrong
        (e.g., wrong species, wrong modality).

  STRICT RULES:
  - Label 0 if ANY hard negative pattern is clearly present.
  - Never infer properties that are not stated in the evidence.
  - Output ONLY valid JSON — no prose before or after.
```

- [ ] **Step 2: Create `neural_search/eval/llm_judge.py`**

```python
"""Optional LLM rubric judge for query-dataset relevance.

Requires: anthropic SDK + ANTHROPIC_API_KEY environment variable.
If either is missing, calling judge_pair() returns None and the caller
should skip gracefully.

Output schema (strict JSON):
{
  "label": 0-3,
  "confidence": 0.0-1.0,
  "rationale": "...",
  "supporting_evidence": ["..."],
  "missing_evidence": ["..."],
  "hard_negative_detected": false
}
"""
from __future__ import annotations

import json
import os
from typing import Any

import yaml

from neural_search.eval.evidence import PairEvidence


_JUDGMENT_SCHEMA = {
    "label": int,
    "confidence": float,
    "rationale": str,
    "supporting_evidence": list,
    "missing_evidence": list,
    "hard_negative_detected": bool,
}


def _default_judgment(rationale: str = "") -> dict:
    return {
        "label": 1,
        "confidence": 0.0,
        "rationale": rationale,
        "supporting_evidence": [],
        "missing_evidence": [],
        "hard_negative_detected": False,
    }


def load_config(config_path: str) -> dict | None:
    """Load judge config. Returns None if file not found."""
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        return None


def _build_prompt(pair: PairEvidence, rubric: str) -> str:
    q = pair.query
    d = pair.dataset
    return f"""## Research Query
Intent: {q.intent}
Query: {q.query_text}
Scientific goal: {q.scientific_goal}
Required modalities: {q.required_modalities}
Required species: {q.required_species}
Known failure modes (hard negatives): {q.hard_negatives}

## Dataset Evidence
Title: {d.title}
Description: {d.description or "Not provided"}
Species: {d.species}
Modalities: {d.modalities}
Brain regions: {d.regions}
Tasks: {d.tasks}
License: {d.license or "Unknown"}
Raw data available: {d.raw_data_available}
Data standards: {d.data_standards}
Metadata completeness: {d.metadata_completeness:.2f}

## Rubric
{rubric}

Respond with ONLY a JSON object matching this schema exactly:
{{"label": <0|1|2|3>, "confidence": <0.0-1.0>, "rationale": "<str>",
  "supporting_evidence": ["<str>", ...], "missing_evidence": ["<str>", ...],
  "hard_negative_detected": <true|false>}}"""


def _parse_judgment(text: str) -> dict | None:
    """Extract and validate the JSON judgment from model output."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Validate required keys and basic types
    for key, expected_type in _JUDGMENT_SCHEMA.items():
        if key not in data:
            return None
        if not isinstance(data[key], expected_type):
            try:
                data[key] = expected_type(data[key])
            except (TypeError, ValueError):
                return None

    data["label"] = max(0, min(3, int(data["label"])))
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    return data


def judge_pair(
    pair: PairEvidence,
    config: dict,
    *,
    n_runs: int | None = None,
) -> dict | None:
    """Judge a single pair. Returns a judgment dict or None if unavailable.

    Raises nothing — all errors are caught and None is returned.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore[import]
    except ImportError:
        return None

    rubric = config.get("rubric", "")
    model = config.get("model", "claude-haiku-4-5-20251001")
    runs = n_runs or config.get("n_runs", 1)

    prompt = _build_prompt(pair, rubric)
    system = (
        "You are a scientific relevance judge. Output ONLY the requested JSON. "
        "Never add prose before or after the JSON object."
    )

    client = anthropic.Anthropic(api_key=api_key)
    judgments: list[dict] = []

    for _ in range(runs):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            parsed = _parse_judgment(text)
            if parsed:
                judgments.append(parsed)
        except Exception:
            continue

    if not judgments:
        return None

    if len(judgments) == 1:
        return judgments[0]

    # Self-consistency: majority vote on label, average confidence
    from collections import Counter
    label = Counter(j["label"] for j in judgments).most_common(1)[0][0]
    confidence = sum(j["confidence"] for j in judgments) / len(judgments)
    # Pick the rationale from the run whose label matched majority
    representative = next(j for j in judgments if j["label"] == label)
    return {**representative, "label": label, "confidence": round(confidence, 4)}
```

- [ ] **Step 3: Create `scripts/eval/judge_candidates.py`**

```python
#!/usr/bin/env python3
"""Optionally judge query-dataset pairs with an LLM rubric judge.

Exits with code 0 and a warning if ANTHROPIC_API_KEY is not set or the
config file is missing — does not block the pipeline.

Usage:
    python scripts/eval/judge_candidates.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --votes artifacts/eval/label_function_votes.jsonl \
        --config configs/judges/rubric_judge_v1.yaml \
        --out artifacts/eval/llm_judgments.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import DatasetEvidence, PairEvidence, QuerySpec
from neural_search.eval.llm_judge import judge_pair, load_config


def _load_pairs(evidence_path: Path) -> list[PairEvidence]:
    pairs = []
    with evidence_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(PairEvidence(
                query_id=row["query_id"],
                record_id=row["record_id"],
                query=QuerySpec(**row["query"]),
                dataset=DatasetEvidence(**row["dataset"]),
                pooled_from=row.get("pooled_from", []),
                min_rank=row.get("min_rank", 1000),
                priority=row.get("priority", "normal"),
            ))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--votes", type=Path, default=None)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(str(args.config))
    if config is None:
        print(f"Warning: judge config not found at {args.config} — skipping LLM judging.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set — skipping LLM judging.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        return

    pairs = _load_pairs(args.evidence)
    max_pairs = config.get("max_pairs") or len(pairs)
    pairs = pairs[:max_pairs]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with args.out.open("w", encoding="utf-8") as out_fh:
        for pair in pairs:
            judgment = judge_pair(pair, config)
            if judgment is None:
                skipped += 1
                continue
            record = {
                "query_id": pair.query_id,
                "record_id": pair.record_id,
                **judgment,
            }
            out_fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"LLM judgments written: {written}, skipped: {skipped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the script handles missing API key gracefully**

```bash
ANTHROPIC_API_KEY="" python scripts/eval/judge_candidates.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --config configs/judges/rubric_judge_v1.yaml \
    --out artifacts/eval/llm_judgments.jsonl
```

Expected output: `Warning: ANTHROPIC_API_KEY not set — skipping LLM judging.`
Exit code: 0

- [ ] **Step 5: Commit**

```bash
git add neural_search/eval/llm_judge.py \
    configs/judges/rubric_judge_v1.yaml \
    scripts/eval/judge_candidates.py
git commit -m "feat(eval): optional LLM rubric judge with graceful fallback"
```
