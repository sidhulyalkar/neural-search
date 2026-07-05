# Paper-Dataset Link Audit Summary (2026-06-24)

## What this is

An LLM-judge pass (Claude Sonnet 4.6) over a 150-row stratified sample (50 DOI-exact, 50
title-fuzzy, 50 not-found) from `artifacts/literature/paper_dataset_links.jsonl`, joined against
dataset titles/descriptions from `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`
(`artifacts/eval/paper_link_audit_joined.jsonl`). Filled template:
`reports/eval/paper_link_audit_template_2026_judged.csv`. As with the finding-extraction audits,
this is an LLM-judge pass, not a human review.

## Headline numbers

| Link type | Sample size | Correct | Notes |
|---|---|---|---|
| `doi_exact` | 50 | **50/50 (100%)** | All NeuroMorpho archive->paper links; species/region/cell-type metadata matches paper topic in every case |
| `title_fuzzy` | 50 | **50/50 (100%)** | Dataset and paper titles are near-verbatim identical in every case (49 exact, 1 partial-but-clearly-correct) |
| `not_found` | 50 | 18 confirmed correct, **8 confirmed false negatives (16%)**, 24 uncertain | See breakdown below |

**DOI-exact and title-fuzzy precision is higher than the project's own conservative estimates**
(`PEER_VALIDATION_PROTOCOL.md` expected 60-85% for title-fuzzy; this sample shows 100%). The
title-fuzzy matcher is finding genuinely near-exact title matches that just lack an extractable
DOI field on the dataset side — it is not meaningfully "fuzzier" than the DOI-exact matcher in
practice, at least in this sample.

## The real finding: a fixable false-negative pattern in `not_found`

8 of 50 `not_found` records have the paper's full citation, title, or DOI **sitting directly in
the dataset's own description field**, unused by the linker:

1. `harvard_dataverse:10.34894_OUS0QP` — description is literally `"Anne Charlotte Trutti, Zsuzsika
   Sjoerds & Bernhard Hommel (2019) Cognitive, Affective, & Behavioral Neuroscience"` plus a
   matching title.
2. `dandi:001370` — description: `"Source data for published work in Carta et al. Nature
   Communications entitled Sex-specific hypothalamic neural projection activity drives caregiving
   in mice"`.
3. `zenodo:17030629` — description contains a **literal DOI link**, `10.1038/s41586-024-07953-5`,
   and the exact Nature paper title. This is the clearest possible miss in the sample: the linker
   had a working DOI handed to it in plain text and didn't use it.
4. `harvard_dataverse:10.34894_W0A8XB` — full author citation in description.
5. `zenodo:13745072` — paper title given verbatim in both dataset title and description.
6. `neurovault:267` — two explicit citations in description (Collins et al. 1995, Mazziotta et al.).
7. `zenodo:15098469` — `"Dataset for the publication: ..."` with full author list and exact title.
8. `harvard_dataverse:10.7910_DVN_MXWL9L` — description references a specific paper title fragment.

**This is mechanically fixable without re-running the fuzzy matcher**: a description-field
citation/DOI scan (regex for `10\.\d{4,9}/\S+` DOI patterns, and a fallback title-fuzzy pass against
the dataset's own `description` text, not just its `title`) would likely resolve most of these —
several literally contain a ready-to-use DOI. This is a clean engineering target, not a research
problem.

**16% is a floor, not a ceiling**: 24/50 records are `uncertain` — well-described, specific
datasets (DANDI, GIN, NeuroVault) with no citation signal visible in the available description, but
no positive evidence of being unpublished either. A few have title text reading exactly like a
paper title (`neurovault:1288` "Branding the brain: A critical review and outlook",
`neurovault:5172` "Maternal Emotion Socialization in Early Childhood Predicts Adolescents'
Amygdala-vmPFC..."), suggesting some fraction of the uncertain group are additional misses that
would require an external literature search to confirm.

**Two NeuroMorpho archives in the not-found sample (`Dwyer`, `Ohgomori`) are suspicious**: every
other NeuroMorpho archive in this audit (all 50 `doi_exact` rows) was correctly linked, since
NeuroMorpho archives are essentially always tied to a specific deposit paper. These two not being
found, against that 100% backdrop, looks more like a missed DOI than a genuine absence of a paper.

## Corpus-scope contamination (separate issue, same pattern as the finding-extraction audits)

Three `not_found` records are not neuroscience data at all:
- `figshare:916090` — cardiac arrhythmia / ventricular sodium current electrophysiology, not brain.
- `harvard_dataverse:10.7910_DVN_VUYB2N` — water/soil biofilter measurements (environmental engineering).
- `zenodo:18684666` — an Uzbek-language pedagogy article about early-childhood cognitive development, not a dataset.

These three are correctly "not found" (no neuroscience paper exists to link, because the record
itself shouldn't be in this corpus), but they confirm the corpus-contamination issue identified in
both 2026 finding-extraction audits is not isolated to literature ingestion — it's also present in
the dataset corpus itself.

## Interpretation against rubric thresholds

Per `PEER_VALIDATION_PROTOCOL.md` §5: DOI-exact expected ≥95% (confirmed: 100%), title-fuzzy
expected 60-85% (actual: 100%, better than expected), not-found false-negative rate "unknown —
audit determines this" (now determined: **≥16%, likely higher**).

## Caveat

LLM-judge pass, not human-reviewed. The 8 confirmed misses are unambiguous (citation/DOI text is
directly quotable from the dataset's own metadata) and don't depend on judgment calls. The 24
`uncertain` rows would need an external literature search (OpenAlex/Google Scholar) to resolve —
deliberately not done here to avoid mixing a real false-negative-rate estimate with unverified web
search guesses.
