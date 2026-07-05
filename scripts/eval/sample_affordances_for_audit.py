"""Affordance validation audit sampler.

Samples datasets for key analysis affordances and produces an audit template
showing required signals found/missing and source evidence.

Target affordances: the 8 most frequent values actually present in the
corpus's `analysis_affordances` field as of 2026-06-24 (see audit note
below) -- population_coding, stimulus_response, calcium_event_detection,
cell_type_characterization, spike_sorting, morphology_analysis,
brain_image_analysis, microscopy_image_registration.

AUDIT NOTE (2026-06-24): the original target list here (q_learning,
choice_decoding, trial_aligned_neural, functional_connectivity,
seizure_detection, speech_decoding, delay_discounting_modeling,
motor_decoding) was drawn from neural_search/analysis_affordances.py's
18-value AFFORDANCE_IDS registry. A live check found ZERO overlap between
that registry and the actual `analysis_affordances` values present in
data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl (990/7,171
records have non-empty affordances, all drawn from a different,
morphology/imaging/electrophysiology-method vocabulary). The registry-based
rule detector (wired into neural_search/enrich_corpus.py) appears to have
never been run against this corpus snapshot, or its output was not carried
into v09 -- see reports/eval/affordance_audit_summary_2026.md.

Usage:
    python scripts/eval/sample_affordances_for_audit.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl"
AUDIT_CSV = ROOT / "reports/eval/affordance_audit_template.csv"
AUDIT_INSTRUCTIONS = ROOT / "reports/eval/affordance_audit_instructions.md"

TARGET_AFFORDANCES = [
    "population_coding",
    "stimulus_response",
    "calcium_event_detection",
    "cell_type_characterization",
    "spike_sorting",
    "morphology_analysis",
    "brain_image_analysis",
    "microscopy_image_registration",
]

# Rough signal requirements per affordance (for audit checklist display)
AFFORDANCE_SIGNALS: dict[str, dict] = {
    "population_coding": {
        "required": ["multi-unit or multi-channel neural recordings", "simultaneous population activity"],
        "preferred": ["sorted spike trains", "task or stimulus labels per trial"],
    },
    "stimulus_response": {
        "required": ["stimulus presentation timestamps/labels", "time-locked neural or behavioral response"],
        "preferred": ["trial structure", "stimulus metadata (type, intensity, duration)"],
    },
    "calcium_event_detection": {
        "required": ["calcium imaging traces (dF/F or raw fluorescence)", "frame timing/sample rate"],
        "preferred": ["ROI/cell segmentation", "behavioral or stimulus alignment"],
    },
    "cell_type_characterization": {
        "required": ["cell-type labels or markers (genetic line, morphology, marker stain)", "per-cell metadata"],
        "preferred": ["electrophysiological or morphological feature data", "multiple identified cell types"],
    },
    "spike_sorting": {
        "required": ["raw or band-passed extracellular voltage traces", "multi-channel/multi-electrode recording"],
        "preferred": ["ground-truth or validated unit labels", "waveform data"],
    },
    "morphology_analysis": {
        "required": ["3D neuronal reconstruction (SWC or equivalent)", "soma/dendrite/axon structure"],
        "preferred": ["cell-type annotation", "species/region metadata"],
    },
    "brain_image_analysis": {
        "required": ["volumetric or 2D brain imaging data", "image metadata (resolution, modality)"],
        "preferred": ["registration to a reference atlas", "segmentation labels"],
    },
    "microscopy_image_registration": {
        "required": ["microscopy image stacks", "spatial/voxel metadata"],
        "preferred": ["reference atlas alignment", "multi-channel or multi-modal imaging"],
    },
}

AUDIT_COLUMNS = [
    "affordance",
    "dataset_id",
    "dataset_title",
    "source",
    "modalities",
    "brain_regions",
    "tasks",
    "species",
    "affordance_confidence",
    "required_signals_found",
    "required_signals_missing",
    "source_evidence",
    "actually_supports_analysis",   # TRUE / FALSE / UNCERTAIN
    "support_type",                 # metadata_only / file_inspected / literature_confirmed
    "false_positive",               # TRUE / FALSE
    "false_negative",               # TRUE / FALSE (if dataset NOT in list but should be)
    "notes",
]

INSTRUCTIONS_TEXT = """# Affordance Validation Audit Instructions

## Purpose
Determine whether Neural Search's affordance labels are scientifically accurate.
This audit makes affordance claims measurable for the whitepaper.

## Template
`reports/eval/affordance_audit_template.csv`

## Columns

| Column | Values | Description |
|---|---|---|
| `actually_supports_analysis` | TRUE / FALSE / UNCERTAIN | Does this dataset genuinely support the claimed affordance? |
| `support_type` | `metadata_only` / `file_inspected` / `literature_confirmed` | How did you determine support? |
| `false_positive` | TRUE / FALSE | Affordance claimed but dataset cannot support it |
| `false_negative` | TRUE / FALSE | Affordance missing but dataset could support it |
| `notes` | free text | Evidence, dataset URL, specific issue |

## Process
1. Review the dataset title, source, modalities, and brain regions.
2. Open the dataset URL if available (DANDI / OpenNeuro / NeuroVault etc.).
3. Check whether required signals (listed in `required_signals_found` column) are present.
4. Set `actually_supports_analysis`: TRUE if the dataset genuinely supports the affordance.
5. Mark `support_type`: was your judgment from metadata only, file inspection, or a linked paper?
6. Mark `false_positive=TRUE` if affordance was incorrectly claimed.

## Target Precision
- ≥80% true positives → affordance labels acceptable for whitepaper claim with caveat.
- 60–80% → report precision; note metadata-only limitations.
- <60% → affordance precision insufficient; re-tune affordance detector.
"""


def load_corpus(n_max: int = 20_000) -> list[dict]:
    records = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_max:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def has_affordance(record: dict, affordance: str) -> bool:
    aff_list = record.get("analysis_affordances") or []
    if isinstance(aff_list, list):
        for a in aff_list:
            if isinstance(a, dict):
                if a.get("affordance_id") == affordance or a.get("affordance") == affordance:
                    return True
            elif isinstance(a, str) and a == affordance:
                return True
    goals = " ".join(record.get("analysis_goals") or []).lower()
    return affordance.replace("_", " ") in goals


def get_affordance_confidence(record: dict, affordance: str) -> str:
    aff_list = record.get("analysis_affordances") or []
    for a in aff_list:
        if isinstance(a, dict):
            if a.get("affordance_id") == affordance or a.get("affordance") == affordance:
                return str(a.get("confidence", ""))
    return ""


def main() -> None:
    if not CORPUS_PATH.exists():
        print(f"Corpus not found: {CORPUS_PATH}")
        return

    print("Loading corpus ...")
    records = load_corpus()
    print(f"  {len(records):,} records loaded")

    rng = random.Random(42)
    rows: list[dict] = []

    for affordance in TARGET_AFFORDANCES:
        matching = [r for r in records if has_affordance(r, affordance)]
        rng.shuffle(matching)
        sample = matching[:15] or records[:5]  # fallback if no affordance matches
        print(f"  {affordance}: {len(matching)} matching -> sampling {len(sample)}")
        signals = AFFORDANCE_SIGNALS.get(affordance, {})
        for r in sample:
            rows.append({
                "affordance": affordance,
                "dataset_id": r.get("source_id", r.get("dataset_id", "")),
                "dataset_title": (r.get("title") or "")[:150],
                "source": r.get("source", ""),
                "modalities": "; ".join(r.get("modalities") or []),
                "brain_regions": "; ".join(r.get("brain_regions") or [])[:150],
                "tasks": "; ".join(r.get("tasks") or []),
                "species": "; ".join(r.get("species") or []),
                "affordance_confidence": get_affordance_confidence(r, affordance),
                "required_signals_found": "; ".join(signals.get("required", [])),
                "required_signals_missing": "",
                "source_evidence": "metadata_only",
                "actually_supports_analysis": "",
                "support_type": "",
                "false_positive": "",
                "false_negative": "",
                "notes": "",
            })

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(AUDIT_INSTRUCTIONS, "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS_TEXT)

    print(f"CSV written          -> {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Instructions written -> {AUDIT_INSTRUCTIONS.relative_to(ROOT)}")
    print(f"  Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
