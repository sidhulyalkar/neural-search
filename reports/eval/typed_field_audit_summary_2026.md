# Typed-Field Extraction Audit Summary (2026-06-24)

## Status: FIXED, verified, and propagated to full-scale relationship mining

All three confirmed bug classes below were fixed the same day:

1. **`negation`**: removed the 7 bare verb-stem patterns (`\binhibit`, `\bsuppress`,
   `\bblocked?\b`, `\bablat`, `\bsilenced?\b`, `\bcounteractivat`, `\bnullif`) that matched
   affirmative descriptions of inhibitory/suppressive phenomena; added `\bno\s+longer\b` and
   `\bbut\s+not\b` to keep catching the genuine negations that previously only fired
   incidentally via the removed patterns.
2. **`frequency_band`**: added a masking pass (`_mask_excluded`) that blanks out known
   molecule/anatomy collocations (`amyloid-beta`, `alpha-lipoic`, `IL-1 beta`, `A-delta fiber`,
   `TNF-alpha`, `gamma-secretase`, etc.) before band matching.
3. **`temporal_pattern`**: same masking mechanism applied to the `\bcyclic\b` -> oscillatory
   collision with "cyclic AMP"/"cyclic GMP".

**Re-extracting the same 32 audited findings with the fixed code**: 12 of 13 originally-FALSE
rows and 2 of 4 originally-PARTIAL rows flipped to correct (one bonus catch the manual audit had
under-counted: row 12, "alpha-lipoic acid", also had a `negation` false positive from "inhibited
capillary cell apoptosis" that wasn't flagged in the original judgment). New tally: **29 TRUE, 1
FALSE (the unrelated `effect_scale` issue, not in scope for this fix), 2 PARTIAL (conceptual
periodicity-scale mismatches, not hard bugs)**.

| | Before | After |
|---|---|---|
| Strict precision | 46.9% | **90.6%** |
| Weighted precision | 53.1% | **93.75%** |

16 new regression tests added (7 negation false-positive cases, 2 negation true-positive
preservation cases, 4 frequency_band exclusion cases, 2 temporal_pattern exclusion cases, plus
updates to 2 pre-existing tests — `test_counteractivation`/`test_inhibit` — that had encoded the
old buggy behavior as expected). 266 tests pass in `test_typed_finding_extractor.py`.

**Propagated to the full pipeline**: discovered in the process that `relationship_builder.py`
expects its input findings file to already carry typed fields, but no committed artifact had ever
actually been produced that way — `findings_tier1_normalized.jsonl` (the file the relationship
builder reads) predated `typed_finding_extractor.py`'s integration into the normalizer and had no
`negation`/`frequency_band`/etc. fields at all. This means `direct_refutation` had been
structurally impossible (always 0), independent of whether the regex bugs above were fixed — a
missing pipeline connection, not a corrupted artifact. Re-ran `normalize_findings.py` against the
full 190,279-finding extraction (now complete) to produce a properly-enriched 190,251-record file,
then re-ran `build_finding_relationships.py` against it for the first time at full scale:

| Metric | Old (12,609-finding snapshot, unenriched) | New (190,251-finding, enriched) |
|---|---|---|
| Finding edges | 194,903 | 200,000 (hit the cap) |
| `opposite_direction` contradictions | 111,186 | 107,642 |
| **`direct_refutation` contradictions** | **0** | **9,833** |
| Base consensus records | 792 | 8,520 |
| Strong consensus (>=0.8, >=3 papers) | 34 | 274 |
| Qualified consensus records | 567 | 9,823 |
| Region co-occurrence edges | 965 | 11,869 |

This is the first time this project has produced a non-zero `direct_refutation` count, and the
first full-scale run of the consensus/qualified-consensus machinery. Strong consensus records now
cite real named regions with double-digit paper counts (left inferior frontal gyrus: 18 papers,
ventral premotor cortex: 12 papers, subthalamic nucleus: 11 papers).

**One more bug found and fixed in the process**: at this larger scale, `region_cooccurrence.jsonl`
contained one row with non-Latin-script region names (`枕部`/`顶叶`, Chinese for
"occipital"/"parietal lobe" — a non-English finding that slipped past upstream filtering) that
normalize to an empty graph-node identifier and crashed the whole KG-ingestion batch. Fixed in
`relationship_kg_builder.py::add_region_cooccurrence_to_graph` to skip non-normalizable region
pairs (with a stats counter and warning log) rather than crash; regression test added.

**Caveat carried forward**: `findings_tier1_normalized.jsonl` is now 271MB (over GitHub's 100MB
limit) and has been gitignored — the small 12,609-record/7.7MB snapshot remains in git history at
commit `e9b0c21` but is no longer the live artifact. The relationship outputs derived from it
(`finding_edges.jsonl` 82MB, `consensus_summaries_qualified.jsonl` 4.2MB, etc.) stay under the
limit and remain tracked. All of the above is still **silver-tier, LLM-extracted evidence** — see
the finding-extraction audits for precision context (82% strict / 90% weighted on the underlying
extraction itself) — not human-validated.

## What this is

An LLM-judge pass (Claude Sonnet 4.6) over a 32-row stratified sample from
`reports/eval/typed_field_audit_sample.jsonl`, judging the four fields the audit's own
instructions prioritize (`negation`, `frequency_band`, `temporal_pattern`, `spatial_frame` — the
ones wired into `kg_builder.py` and consulted by `relationship_builder.py`'s supports/contradicts
logic), with the four context fields reviewed more lightly. Filled template:
`reports/eval/typed_field_audit_template_2026_judged.csv`.

**Before judging, this run also fixed a real performance bug**: `sample_typed_fields_for_audit.py`
calls `enrich_finding()` (27-field rule-based extractor) on all 190,279 findings to stratify the
sample, and that took 30-45+ minutes (still climbing when killed) because
`neural_search/literature/typed_finding_extractor.py`'s `_any()`/`_first()` helpers called
`re.search(pattern_string, text, re.IGNORECASE)` directly — recompiling from scratch on every call
once the ~560 distinct patterns across 28 field collections exceeded CPython's shared, process-wide
512-entry regex cache. Added a dedicated `@lru_cache`-memoized compile wrapper scoped to this
module; the same run completed in 5m40s afterward. 247 existing tests still pass.

## Headline numbers

| Metric | Value |
|---|---|
| Total judged | 32 |
| TRUE | 15 (46.9%) |
| PARTIAL | 4 (12.5%) |
| FALSE | 13 (40.6%) |
| **Strict precision** | **46.9%** |
| **Weighted precision (TRUE + 0.5×PARTIAL)** | **53.1%** |

Per the audit's own rubric (`<60% on a field -> stop promoting that field into graph edges until
fixed`), **this sample fails the threshold** — a meaningfully different and more concerning result
than the finding-extraction or affordance audits, both of which cleared 80%+.

## The dominant failure: systematic `negation` false positives (9 of 13 FALSE rows)

`negation` is populated (`TRUE`) in 16 of the 32 sampled findings. Of those, roughly 6-7 are correct
(genuine syntactic negation: "did not affect", "no longer observed", "absence of an adapting
stimulus", "not found") — but **at least 7, plausibly more**, are false positives where a content
word describing an inhibitory/suppressive *neuroscience phenomenon* triggered the same pattern that
was meant to catch negated *claims*:

- `"Suppression of posterior alpha oscillations was preceded by..."` — "Suppression" is the
  finding's subject (an affirmative description of what was observed), not a negated claim.
- `"Baclofen blocked the frequency-dependent depression of EPSCs..."` — the drug's blocking effect
  *is* the finding being reported, affirmatively.
- `"...suppressed abnormal oscillatory activity in the basal ganglia..."` — same pattern: DBS's
  suppressive effect is the (affirmative) result.
- `"Phasic ACh application generates biphasic inhibitory followed by excitatory responses"` — a
  real physiological response being described, not a negation.
- `"Most cells inhibited during SPW (80%) fired rhythmically..."` — "inhibited" describes which
  cells are being talked about; the actual finding ("fired rhythmically and phase-locked") is
  affirmative.
- `"Increases in the theta range...were equally high for inhibition and change..."` — "inhibition"
  here is a *task condition name* (alongside "change" and "action activation"), not suppressed
  neural activity.
- `"Blockade of the tonic inhibitory GABAergic input...increased the release of dopamine"` —
  "inhibitory" describes the input being blocked; the actual finding is an affirmative increase.

The root cause: `_NEGATION_PATTERNS` in `typed_finding_extractor.py` includes bare verb-stem
patterns (`\binhibit`, `\bsuppress`, `\bblocked?\b`, `\bablat`, `\bsilenced?\b`, `\bcounteractivat`,
`\bnullif`) that match these words regardless of grammatical role — whether they describe the
*manipulation/phenomenon itself* (which is neuroscience content, often affirmatively reported) or
*actually negate a result*. This is a fundamentally different failure mode from a simple miss: it's
a category error baked into the pattern list design.

**This has a direct, already-realized downstream consequence.** `negation` is the field
`relationship_builder.py`'s `contradiction_subtype` logic (shipped earlier today, see
`reports/strategy/2026-06-23-ir-evaluation-rigor.md`-adjacent Phase 6b work) consults to distinguish
`direct_refutation` from `opposite_direction` contradictions. The `finding_edges.jsonl` regenerated
earlier in this session (194,903 edges, 111,186 `contradicts`) was built with this same negation
detector. **Some unknown fraction of those contradiction edges are likely mislabeled** as a result
of this false-positive pattern — findings that affirmatively report an inhibitory/suppressive
neuroscience effect may have been miscategorized relative to findings they're compared against.
This doesn't invalidate the relationship-mining infrastructure, but it does mean the
`contradiction_subtype` breakdown reported earlier today (`opposite_direction: 111,186,
direct_refutation: 0`) should be treated as provisional pending a fix to the negation patterns,
not as a clean result.

One missed-negation case in the opposite direction was also found: `"correlated with theta but not
beta oscillatory amplitudes"` — the explicit `"but not X"` contrast was never flagged, because the
pattern list only catches specific phrasings (`"not affected"`, `"not observed"`, etc.), not the
general `"X but not Y"` construction.

**Recommended fix**: remove or sharply narrow the bare-verb-stem patterns
(`\binhibit`, `\bsuppress`, `\bblocked?\b`, `\bablat`, `\bsilenced?\b`, `\bcounteractivat`,
`\bnullif`) — they account for essentially all the false positives in this sample. The remaining
patterns (explicit `"not X"`, `"failed to"`, `"absence of"`, `"unchanged"`, `"no significant"`)
performed correctly on every row where they fired.

## Second failure: `frequency_band` collides with Greek-letter molecule/anatomy names (4 of 13 FALSE rows)

`\balpha\b`, `\bbeta\b`, `\bdelta\b` (and presumably `\bgamma\b`) match any standalone occurrence of
these words — including when they're part of molecular or anatomical nomenclature, not an EEG/LFP
frequency band:

- `"IL-1 beta mRNA levels"` -> `frequency_band: beta` (it's the cytokine interleukin-1β)
- `"alpha-lipoic acid administration"` -> `frequency_band: alpha` (a dietary supplement/antioxidant)
- `"A delta fibre stimulation"` -> `frequency_band: delta` (a peripheral nerve fiber class, Aβ/Aδ/C)
- `"Amyloid-beta (Aβ) exposure"` -> `frequency_band: beta` (a protein)

All four are confirmed, unambiguous false positives in just 32 sampled rows — Greek-letter overlap
with molecular biology nomenclature (interleukins, amyloid-beta, alpha-synuclein, delta receptors,
gamma-secretase, etc.) is common enough in neuroscience abstracts that this is likely a substantial,
systematic source of `frequency_band` noise across the full 190K-finding corpus, not a rare edge
case. **Recommended fix**: require frequency-band words to appear near an oscillation/power/band
context word (e.g. `\b(theta|alpha|beta|delta|gamma)\b.{0,20}\b(power|oscillation|band|wave|rhythm|amplitude|activity|frequency)\b`
or the reverse order) rather than matching the bare Greek-letter word alone.

## Minor findings

- `temporal_pattern=oscillatory` was applied to a cell-fate finding with no temporal content at all
  (false positive, unrelated to negation/frequency issues), and to a behavioral rhythm-discrimination
  task and a multidien (multi-day-scale) seizure-timing finding (both defensible-but-stretched uses
  of "oscillatory" for non-EEG-band periodicity).
- One `effect_scale=modest` was applied to a finding explicitly describing "moderate to severe"
  outcomes — modest and severe are close to opposite.

## Caveat

LLM-judge pass, not human-reviewed. Judgments above are conservative in one direction: several rows
were flagged for their most salient error only (e.g. a row with both a `frequency_band` molecule
collision and an `inhibit`-triggered `negation` false positive may only have `frequency_band` listed
in `wrong_fields`), so the true negation false-positive count in this sample is likely higher than
the 7 explicitly counted above. The 46.9%/53.1% precision figures should be read as a credible upper
bound on field-level correctness, not a generous estimate.
