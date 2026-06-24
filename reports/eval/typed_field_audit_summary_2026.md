# Typed-Field Extraction Audit Summary (2026-06-24)

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
