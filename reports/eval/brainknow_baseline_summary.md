# BrainKnow-Style Co-Occurrence Baseline

_Generated: 2026-06-22T05:52:11.294436+00:00_

## Method

Same-sentence concept co-occurrence following Wang et al. (2024) BrainKnow.
Source: `artifacts/literature/findings_tier1_ollama.jsonl`

| Parameter | Value |
|---|---|
| Input records | 122,544 |
| Concept vocabulary | 177 patterns → 177 canonical concepts (regions, signals, tasks, cell types, neuromodulators, disorders, species, methods, models) |
| Unique concepts found | 175 |
| Undirected weighted edges (≥2 co-occurrences) | 4,993 |

## Top Concepts

| Concept | Count |
|---|---|
| human | 54,957 |
| mouse | 29,189 |
| rat | 15,954 |
| hippocampus | 12,388 |
| fMRI/BOLD | 8,475 |
| prefrontal cortex | 6,351 |
| primate | 4,771 |
| striatum | 4,360 |
| visual cortex | 3,884 |
| EEG | 3,883 |
| learning | 3,683 |
| Alzheimer's disease | 3,228 |
| motor cortex | 3,179 |
| amygdala | 3,082 |
| pharmacology | 2,949 |
| cerebellum | 2,746 |
| glutamate | 2,527 |
| thalamus | 2,462 |
| lesion/inactivation | 2,272 |
| dopamine | 2,254 |

## Top Co-Occurring Pairs

| Pair | Weight |
|---|---|
| fMRI/BOLD ↔ human | 7,879 |
| hippocampus ↔ mouse | 5,486 |
| human ↔ prefrontal cortex | 3,991 |
| hippocampus ↔ rat | 3,288 |
| EEG ↔ human | 3,261 |
| hippocampus ↔ human | 2,840 |
| Alzheimer's disease ↔ human | 2,095 |
| human ↔ striatum | 2,078 |
| human ↔ motor cortex | 1,998 |
| human ↔ temporal cortex | 1,755 |
| anterior cingulate ↔ human | 1,617 |
| human ↔ visual cortex | 1,568 |
| amygdala ↔ human | 1,565 |
| human ↔ learning | 1,533 |
| fMRI/BOLD ↔ prefrontal cortex | 1,453 |
| attention ↔ human | 1,444 |
| human ↔ parietal cortex | 1,411 |
| CA1 ↔ hippocampus | 1,384 |
| human ↔ insula | 1,345 |
| microglia ↔ mouse | 1,343 |

## What This Baseline Cannot Answer (vs. Neural Search Typed Graph)

These are queries that co-occurrence alone cannot resolve,
illustrating the motivation for Neural Search's typed relation graph:

- Which dataset can I download to test this claim? (no dataset nodes)
- Does this paper report increase or decrease? (unsigned edges)
- Did co-occurrence result from A inhibiting B vs. A activating B? (untyped)
- What analysis can I run on this dataset? (no affordance nodes)
- Is this evidence strong or a null result? (no negation modeling)
- Does this dataset have raw or processed data? (no data format nodes)
- Was the finding replicated or contradicted? (no polarity)
- Is this a hard negative for my query? (no hard-negative inference)
- What species / frequency band / temporal pattern? (not extracted into graph)

## BrainKnow Comparison

| Dimension | BrainKnow | This Baseline | Neural Search |
|---|---|---|---|
| Scale | 3.6M edges / 1.8M papers | ~edges from findings | typed graph 31,920 edges |
| Edge type | undirected co-occurrence | undirected co-occurrence | typed (supports/contradicts/records/affords…) |
| Dataset nodes | no | no | yes (7,171 datasets) |
| Polarity/negation | no | no | partially implemented |
| Analysis affordances | no | no | yes (21 types) |
| Frequency/temporal | no | no | planned (Task 8) |
| Retrieval target | concepts | concepts | reusable datasets |