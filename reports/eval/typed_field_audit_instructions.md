# Typed Field Extraction Audit Instructions

## Purpose
Evaluate the precision of the 27 rule-based typed fields in
`neural_search/literature/typed_finding_extractor.py`, focusing first on the
four fields wired into the knowledge graph and relationship builder:
`negation`, `frequency_band`, `temporal_pattern`, `spatial_frame`.

## File to Fill
`reports/eval/typed_field_audit_template.csv`

## Columns

| Column | Description |
|--------|-------------|
| `finding_id` / `paper_id` | Identifiers — do not change |
| `finding_text` | The extracted finding sentence |
| `negation` | True if the extractor flagged this as a negated finding |
| `frequency_band` | Extracted frequency band(s), e.g. theta, gamma |
| `temporal_pattern` | Extracted temporal pattern(s), e.g. transient, oscillatory |
| `spatial_frame` | Extracted spatial frame, e.g. local, inter_regional |
| `condition` / `effect_scale` / `behavioral_measure` / `population_type` | Context fields — review only if time permits |
| `human_correct` | Your judgment: `TRUE` / `FALSE` / `PARTIAL` |
| `wrong_fields` | If not TRUE, which column(s) were wrong (space-separated column names) |
| `notes` | Free text |

## Guidelines

1. Read `finding_text` and judge whether each populated field is actually
   correct for that sentence — not whether it is plausible in general.
2. An empty field is not necessarily wrong; only mark `wrong_fields` for
   fields that are populated but incorrect, or that should have matched but
   didn't (note the latter in `notes`, since `wrong_fields` is for false
   positives the column scheme can track).
3. Pay special attention to `negation` — this field gates whether a finding
   counts as supporting evidence in `relationship_builder.py`. A missed
   negation (should be True but is False) is the most consequential error
   type to flag in `notes`.

## Target
- 80%+ precision on the four primary fields -> safe to keep wiring them into
  the graph as-is.
- 60-80% -> usable but flag the dominant failure pattern for a targeted
  lexicon fix before expanding further (see
  docs/superpowers/plans/2026-06-22-typed-field-coverage-relationship-expansion.md).
- <60% on a field -> stop promoting that field into graph edges until fixed.

## Contact
Sid: sid.soccer.21@gmail.com
