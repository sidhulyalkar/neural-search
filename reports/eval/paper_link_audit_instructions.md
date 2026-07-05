# Paper-Dataset Link Audit Instructions

## Purpose
Validate whether the paper-dataset linker correctly associates papers with datasets.
This audit determines whether linker precision is sufficient for whitepaper claims.

## Files
- Template: `reports/eval/paper_link_audit_template.csv`
- Links source: `artifacts/literature/paper_dataset_links.jsonl`

## Columns

| Column | Description | Values |
|---|---|---|
| `link_type` | How the link was found | `doi_exact`, `title_fuzzy`, `not_found` |
| `dataset_record_id` | Neural Search dataset ID | — |
| `dataset_title` | Dataset title | — |
| `paper_doi` | Paper DOI (if available) | — |
| `paper_title` | Paper title | — |
| `paper_year` | Publication year | — |
| `match_method` | Matching strategy used | — |
| `confidence` | Linker confidence score | — |
| `link_correct` | Your judgment | `TRUE` / `FALSE` / `UNCERTAIN` |
| `error_type` | If not TRUE | `wrong_paper`, `no_relationship`, `ambiguous`, `none` |
| `notes` | Free text | any |

## For "not_found" rows
These are dataset records where the linker found **no** paper link.
Assess whether you can find a paper for this dataset manually.
If yes, mark `link_correct=FALSE` (false negative) and add the paper in `notes`.

## Target Precision
- DOI-exact: expect ≥95% precision.
- Title-fuzzy: expect 60–85% precision.
- Not-found recall: measure how many have discoverable papers.
