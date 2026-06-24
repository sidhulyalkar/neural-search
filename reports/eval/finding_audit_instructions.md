# Finding Extraction Audit Instructions

## Purpose
Evaluate the precision of LLM-extracted findings from `artifacts/literature/findings_tier1_ollama.jsonl`.
This audit determines whether extraction quality is sufficient to cite in the Neural Search whitepaper.

## File to Fill
`reports/eval/finding_audit_template.csv`

## Columns

| Column | Description | Values |
|--------|-------------|--------|
| `finding_id` | Unique ID — do not change | — |
| `paper_id` | OpenAlex paper ID — do not change | — |
| `finding_text` | Extracted finding sentence | — |
| `regions` | Extracted brain region(s) | — |
| `tasks` | Extracted task(s) | — |
| `modalities` | Extracted modality/modalities | — |
| `species` | Extracted species | — |
| `result_direction` | increase / decrease / no_change / correlation / mixed | — |
| `confidence` | Extractor confidence [0–1] | — |
| `human_correct` | Your judgment | `TRUE` / `FALSE` / `PARTIAL` |
| `error_type` | If not TRUE, what is wrong | `region_wrong`, `task_wrong`, `direction_wrong`, `hallucinated`, `species_wrong`, `none` |
| `notes` | Free text | any |

## Guidelines

1. Open the paper (use `paper_id` to look up on OpenAlex: https://openalex.org/works/<paper_id>).
2. Find the relevant passage in the abstract or methods.
3. Assess whether:
   - The finding text faithfully represents what the paper reports.
   - The brain region(s) are correct and specific.
   - The task is correctly identified.
   - The result direction (increase/decrease/no_change) is correct.
   - The species is correct.
4. Mark `human_correct`:
   - `TRUE` — all fields accurate.
   - `PARTIAL` — some fields correct, one minor error.
   - `FALSE` — major error or hallucination.
5. If `FALSE` or `PARTIAL`, fill `error_type` with the primary error category.
6. Record any notes in the `notes` column.

## Target
- 80%+ precision → findings acceptable for whitepaper claims with caveats.
- 60–80% → report precision with specific failure modes; use as audit queue only.
- <60% → extraction quality insufficient; re-run with improved prompt.

## Contact
Sid: sid.soccer.21@gmail.com
