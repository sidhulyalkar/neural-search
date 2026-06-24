# LLM-Judge Audit Summary: Tier-1 Ollama Finding Extraction (Post-Completion, 2026-06-23)

## What this is

A second **LLM-judge pass** (Claude Sonnet 4.6), run after the tier-1 finding extraction job
reached 100% completion (255,940/255,940 papers checkpointed, 190,279 findings extracted —
up from the 16,150-papers-processed / 12,050-findings snapshot judged in the original
2026-06-22 audit, `finding_audit_llm_judge_summary.md`). This audit answers the question the
first one couldn't: does extraction precision hold up at full scale, or does it degrade as the
run processes the long tail of the corpus?

Source data: `reports/eval/finding_audit_sample.jsonl` (100 rows, freshly stratified-sampled from
the complete 190,279-finding file, seed=2026), joined with paper title + abstract from
`data/corpus/normalized/openalex_neuro/tier1_batch_*.jsonl` (`artifacts/eval/finding_audit_joined_2026.jsonl`).
Filled template: `reports/eval/finding_audit_template_2026_judged.csv`. As before, this is an
LLM-judge pass, not a human review — see the caveat at the bottom.

**Methodology note:** abstracts were judged from a 700-character preview per row for the primary
pass; full abstracts were pulled for any row where the preview looked ambiguous, contradictory,
or suspicious (6 rows: #1, #4, #7, #19, #23, #36). Rows accepted as TRUE on a plausible-but-truncated
preview were not independently re-verified against the full text — a small number of additional
errors could exist in that subset that a full-text pass would catch.

## Headline numbers

| Metric | 2026-06-22 (16K papers / 12K findings) | 2026-06-23 (255,940 papers / 190,279 findings, complete) |
|---|---|---|
| Total judged | 100 | 100 |
| TRUE (fully correct) | 88 (88.0%) | 82 (82.0%) |
| PARTIAL (one minor error) | 8 (8.0%) | 16 (16.0%) |
| FALSE (major error / hallucination) | 4 (4.0%) | 2 (2.0%) |
| **Strict precision (TRUE only)** | **88.0%** | **82.0%** |
| **Weighted precision (TRUE + 0.5×PARTIAL)** | **92.0%** | **90.0%** |

**Precision held up at full scale** — both strict and weighted precision dropped modestly (88%→82%,
92%→90%) but stayed well above the project's 80% "acceptable for whitepaper claims with caveats"
threshold (`PEER_VALIDATION_PROTOCOL.md` §4). The drop is consistent with judging on a fresh
independent 100-row sample rather than a regression in extraction quality — the FALSE rate actually
*halved* (4%→2%), while more borderline PARTIAL cases were caught this time, partly because this
pass deliberately pulled full abstracts to resolve ambiguous cases (which surfaced 2 confirmed
region-hallucination PARTIALs that a preview-only pass might have missed) and partly because it
flagged a genuinely new failure category not seen in the first 100-row sample (see below).

## Error type breakdown (18 non-TRUE rows)

| error_type | count |
|---|---|
| region_wrong | 6 |
| direction_wrong | 4 |
| species_wrong | 4 |
| hallucinated | 2 |
| other (method-description-as-finding) | 2 |

**`region_wrong` is now the dominant failure mode** (6/18, plus the 2 `hallucinated` rows are also
region errors — 8/18 region-related total if grouped together), where the 2026-06-22 audit found
`direction_wrong` dominant (6/12). Two distinct region-error subtypes emerged:

1. **Inferred-not-stated** (4 rows: #4, #23, #71-partial, #78): a plausible-sounding anatomical
   region tagged based on the paper's general topic or sensory modality (AD → hippocampus,
   olfactory task → olfactory cortex, touch study → S1) even though the abstract never names that
   structure. This is the same failure shape the 2026-06-22 audit's `region_wrong` example showed
   (midoccipital cortex misattributed to the wrong processing phase) — inference outrunning text
   evidence.
2. **Malformed/non-anatomical values** (3 rows: #71, #89, #97 — new in this pass): the region field
   populated with something that isn't a brain region at all — a genomic locus (`chromosome 19`),
   a peptide epitope name (`mbp(111-129)`), or a literal comparative phrase concatenated as if it
   were a place name (`higher than primary visual cortex`). Two of these (#89, #97) are the
   FALSE-rated hallucinations. This subtype wasn't documented in the first audit and is a concrete,
   fixable target: a post-extraction validator that rejects region values containing digits-after-letters
   patterns (`chromosome 19`), parenthesized residue ranges (`mbp(111-129)`), or comparative words
   (`higher than`, `lower than`, `more than`) would catch all three of these mechanically, no LLM
   re-judging required.

**New failure category: method description extracted as finding** (2 rows: #28, #33). Sentences
like *"Ss' sensitivities (d's) for discriminating between items... were measured"* or *"Perforated-patch
... recordings were necessary to accurately study..."* are methodological statements, not results —
there is no actual finding here, yet the extractor produced one (with a default `direction` value,
since there's no real direction to extract). This is a distinct failure from `direction_wrong`
(misapplying a direction to a real-but-non-directional finding) — here there's no finding at all.
Worth a dedicated filter pass: reject candidate findings whose verb phrase is purely methodological
("was measured", "were necessary to", "was used to assess") rather than result-bearing.

**`species_wrong` (4 rows, same rate as before — 4/12 in 2026-06-22, 4/18 here)**: all four are
omissions or vague placeholders rather than wrong-species hallucinations this time (species field
empty for clearly-human subjects in #50, #67, #94; generic `mammals` placeholder for a genuinely
unspecified animal model in #19 — the one borderline case, arguably acceptable since the abstract
truly doesn't name a species). No outright wrong-species hallucination (e.g. last time's
"PET-scanned children mislabeled mouse") recurred in this sample.

**`direction_wrong` (4 rows)**: two are the same root cause as 2026-06-22 — a non-directional
mechanistic/architectural finding forced into `no_change` (#6, #65) rather than tagged `mechanism`
(which the extractor *does* use correctly elsewhere in this same sample — #54, #93 shows the
schema is capable of the right answer, just inconsistently applied). One is an oversimplified
correlation claim that erases a non-monotonic pattern stated in the abstract (#57). One is a
mixed-direction finding (ATP down, lipid peroxidation up) force-tagged as a single `decrease`
instead of `other` (#93) — again, the schema has an `other` value and uses it correctly elsewhere
in this sample (#55, #56, #58, #61, #86), so this is inconsistent application, not a missing capability.

**Corpus/domain contamination** (confirmed in #1, #7; suspected in several non-neuroscience but
textually-accurate rows like #18, #31, #41, #42, #45, #74): the pipeline continues to admit clearly
non-neuroscience papers (adolescent sexual-health behavior, kidney fibrosis, orthopedic biomechanics,
bacterial ion channels, bone collagen biology) into the findings corpus, consistent with the single
semiconductor-physics example flagged in the 2026-06-22 audit. This is now confirmed as a recurring
pattern, not a one-off — the topic/domain filter recommendation from the first audit remains open
and is higher-priority given the confirmed recurrence.

## Interpretation against rubric thresholds

- Rubric: 80%+ → acceptable for whitepaper claims with caveats; 60–80% → caveated/audit-queue
  only; <60% → insufficient, re-extract.
- **Result: 82.0% strict / 90.0% weighted — clears the 80% bar**, same conclusion as the
  2026-06-22 audit, now confirmed at full corpus scale (190,279 findings, 255,940 papers, complete).
  Findings remain acceptable for citation in the whitepaper **with caveats**:
  - Treat extracted `regions` as needing a sanity check, especially for inferred-not-stated cases
    and for any value that doesn't look like a real anatomical term (genomic loci, gene/peptide
    names, comparative phrases). This is the single most actionable, mechanically-fixable item
    from this pass.
  - Treat `result_direction` as unreliable for mechanistic, architectural, or genuinely mixed-direction
    findings — the schema supports `mechanism` and `other` correctly in places, but applies them
    inconsistently.
  - The pipeline should add (a) a topic/domain filter upstream of extraction (recurring, confirmed
    issue, not a one-off) and (b) a finding-vs-methods-statement filter downstream.

## Confirmed-by-full-text examples (this pass specifically verified against untruncated abstracts)

1. **Region hallucination, AD visuospatial study** — `paper:openalex:W2067587974:f1`. Finding:
   *"These impairments were associated with poorer performance on the Money Road Map test of
   spatial navigation."* Extracted `region: hippocampus`. Full abstract never mentions hippocampus
   — the actual reported correlates are "posterior cortical atrophy and impaired visual motion
   processing." Region inferred from AD-domain prior knowledge, not text evidence.

2. **Corpus contamination + region hallucination, non-neuroscience paper** — `paper:openalex:W4246507697:f0`.
   Finding about condom use among adolescents, extracted `region: san francisco` (a city). Full
   abstract confirms zero neuroscience content; this is a public-health behavior paper that should
   never have entered the corpus.

3. **Confirmed correct on full-text check (a near-miss for a false flag)** — `paper:openalex:W4366981333:f1`.
   Finding: *"Knocking out Piezo1 in CEA neurons showed a significant reduction of response under
   ultrasound stimulation."* The 700-character preview only showed a different experiment in the
   same paper (motor-cortex knockout), making this look like a region mismatch at first read. The
   full abstract confirms the CEA experiment is described later, verbatim. Worth noting as a
   methodology lesson: single-experiment papers with multiple knockout/region conditions need full
   text, not preview windows, to judge correctly — a real risk for any future audit pass run on
   truncated previews alone.

4. **Missing species despite explicit statement** — `paper:openalex:W2296780006:f1`. Finding about
   TGF-beta in glomeruli; full abstract explicitly studies "the rabbit" throughout, but the
   extracted `species` field is empty.

## Caveat

Same as the 2026-06-22 pass: this audit was performed by an LLM (Claude Sonnet 4.6) reading
abstracts and judging the Ollama extractor's output — no human has reviewed either the source
papers or this judge's calls. The four method-vs-result and direction-consistency findings above
are reasonably objective (the schema's own `mechanism`/`other` values prove it can do better), but
several "plausible continuation" TRUE verdicts in this pass were given on truncated 700-character
previews without independent full-text verification (see Methodology note above) and could be
revised by a full-text or human pass. If cited externally, label as **LLM-judge precision, pending
human audit** — not a validated human accuracy figure.
