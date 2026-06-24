# LLM-Judge Audit Summary: Tier-1 Ollama Finding Extraction

## What this is

This is an **LLM-judge pass** (Claude Sonnet 4.6) standing in for human review of 100 sampled
findings extracted from paper abstracts by a local Ollama model (`qwen2.5:7b-instruct-q4_K_M`)
in the literature-mining pipeline. No human has reviewed this audit queue yet. The judge
(Claude) is a different model than the extractor (Ollama/Qwen), which gives some independence,
but **this is not a substitute for human review** — see the caveat at the bottom.

Source data: `artifacts/eval/finding_audit_joined.jsonl` (100 rows, each finding joined with its
source paper's title + abstract). Full judgments: `artifacts/eval/llm_judgments.jsonl`.
Filled template: `reports/eval/finding_audit_template_llm_judged.csv`. The original blank
template (`reports/eval/finding_audit_template.csv`) was left untouched for an eventual human
audit pass.

## Headline numbers

| Metric | Value |
|---|---|
| Total judged | 100 |
| TRUE (fully correct) | 88 (88.0%) |
| PARTIAL (one minor error) | 8 (8.0%) |
| FALSE (major error / hallucination) | 4 (4.0%) |
| **Strict precision (TRUE only)** | **88.0%** |
| **Weighted precision (TRUE + 0.5×PARTIAL)** | **92.0%** |

## Error type breakdown (12 non-TRUE rows)

| error_type | count |
|---|---|
| direction_wrong | 6 |
| species_wrong | 4 |
| hallucinated | 1 |
| region_wrong | 1 |

**Dominant failure mode: `result_direction` mischaracterization.** Half of all non-TRUE rows
(6/12) involve the extractor assigning a directional label (most often `increase`) to findings
that are not really about magnitude/direction at all — e.g., decoding/predictability results,
mechanistic conclusions ("supramodal vs. sensory-specific"), or comparative null results. The
`result_direction` taxonomy (`increase`/`decrease`/`no_change`/`correlation`/`mechanism`/`other`)
appears under-used in favor of defaulting to `increase`, and `no_change` is sometimes used
for "this is a positive mechanistic finding" rather than "this is a literal absence of effect."

**Second-largest: `species_wrong` (4/12).** All four are cases where the species field doesn't
match the clearly human (or clearly non-human) subjects described in the abstract — including
one outright hallucination (PET-scanned children mislabeled "mouse") and one omission (blank
species for an obviously human stroke-patient study).

**One `hallucinated` (domain-extraction) error**: a semiconductor-physics paper (gate oxide
breakdown percolation modeling) was extracted into the findings corpus and given a fabricated
"region" field (`oxide`), even though the paper has nothing to do with neuroscience. This
suggests the corpus/pipeline upstream of the LLM extractor occasionally feeds it off-topic
papers, and the extractor doesn't reject them.

**One `region_wrong`**: a defensible-but-unsupported attribution of a maintenance-period EEG
effect to a region (midoccipital cortex) that the abstract only explicitly links to a different
processing phase (encoding).

## Interpretation against rubric thresholds

- Rubric: 80%+ → acceptable for whitepaper claims with caveats; 60–80% → caveated/audit-queue
  only; <60% → insufficient, re-extract.
- **Result: 88.0% strict / 92.0% weighted — clears the 80% bar.** Findings from this extraction
  run are acceptable for citation in the whitepaper **with caveats**, specifically:
  - Treat `result_direction` as unreliable for findings whose nature is mechanistic, comparative,
    or about predictability/decoding rather than simple magnitude change. Spot-check before
    citing any claim that hinges on directionality.
  - Treat `species` as needing a sanity check, especially for studies that mix human clinical
    measures with phrasing that could be mistaken for animal-model boilerplate.
  - The pipeline should add a topic/domain filter upstream of extraction — at least one
    extracted "finding" came from a non-neuroscience paper.

## Worst failures (concrete examples)

1. **Species hallucination on human pediatric data** — `paper:openalex:W2014377355:f0`
   Finding: *"The autistic group had decreased glucose metabolism in the lateral temporal gyri
   bilaterally compared to the mentally-retarded nonautistic group."* Extracted `species: mouse`.
   Abstract: *"Children with TSC and intractable epilepsy underwent MRI as well as PET scans...
   autistic (OABC < 70; n = 9), mentally-retarded nonautistic..."* — explicitly a pediatric human
   clinical study (PET/MRI, Vineland Adaptive Behavior Scale). No animal model involved.

2. **Off-topic paper hallucinated into the neuroscience corpus** — `paper:openalex:W2059074447:f1`
   Finding: *"The effective defect 'size' of about 3 nm is obtained from the thickness dependence
   of the breakdown distributions."* Extracted `region: oxide`. The source paper, *"Percolation
   models for gate oxide breakdown,"* is semiconductor device physics — it has no brain region,
   no species, no neuroscience content whatsoever. The extractor faithfully transcribed the
   sentence but should never have produced a "finding" with brain-region semantics from this
   paper; this is a pipeline/corpus contamination issue rather than a misreading of the text.

3. **Species hallucination on a human gambling-task fMRI study** — `paper:openalex:W2144547056:f1`
   Finding: *"Women had greater activation in the left DLPFC, left medial frontal gyrus and
   temporal lobe during the task."* Extracted `species: mouse`. Abstract: *"When men and women
   were examined separately, men activated extensive regions of the right lateral OFC..."* —
   this is unambiguously a human sex-differences neuroimaging study (Iowa Gambling Task).

4. **Direction field misused for a non-directional astrocyte mechanism finding** —
   `paper:openalex:W2911036111:f2` Finding: *"Dysfunctional chaperone-mediated autophagy (CMA)
   and impaired macroautophagy were identified in PD astrocytes."* Extracted `species: other`,
   `region: ventral midbrain`. Abstract describes **human iPSC-derived astrocytes and neurons**
   from PD patients and healthy controls (`"generated induced pluripotent stem cell-derived
   astrocytes and neurons from familial mutant LRRK2 G2019S PD patients and healthy
   individuals"`) — species should be human, not "other." The CMA/autophagy defect is described
   for astrocytes generally/in culture, not specifically localized to "ventral midbrain" (that
   region pertains to the separately-described dopaminergic neuron degeneration, a different
   part of the same abstract).

5. **Directional overclaim on a "modulation, direction unspecified" finding** —
   `paper:openalex:W2069879324:f0` Finding: *"The orbitofrontal cortex shows increased activity
   when selecting a devalued action during goal-directed learning."* Abstract: *"neural activity
   in one brain region in particular, the orbitofrontal cortex, showed a strong modulation in its
   activity during selection of a devalued compared with a nondevalued action"* — the abstract
   explicitly avoids stating the direction of the OFC modulation, but the extractor invented
   "increased."

## Caveat

This audit was performed entirely by an LLM (Claude Sonnet 4.6) reading abstracts and judging
the Ollama extractor's output — there has been **no human-in-the-loop verification** of either
the source papers beyond their abstracts, or of this judge's own calls. Borderline calls
(especially several of the `direction_wrong` PARTIALs, which hinge on a subjective read of
whether a result is "directional" at all) could reasonably be re-classified by a human reviewer.
If this summary or its precision numbers are cited externally (e.g., in the whitepaper), they
should be labeled as **LLM-judge precision, pending human audit**, not as a validated human
accuracy figure.
