# Affordance Validation Audit Instructions

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
