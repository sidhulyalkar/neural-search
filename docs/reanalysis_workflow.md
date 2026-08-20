# Reanalysis as a first-class workflow

Neural Search treats dataset reuse as more than a retrieval result. The reanalysis workbench asks a second question after a dataset is found:

> Given the signals and metadata that appear to exist, which analysis families are feasible, which concrete methods fit those families, what is missing, what precedent exists, and what would still require scientific judgment?

## Interfaces

CLI:

```bash
neural-search reanalysis DEMO_VISUAL_DECISION_NEUROPIXELS
```

API:

```text
GET /api/reanalysis/{dataset_id}?limit=12
```

Web:

```text
/reanalysis
/reanalysis/{dataset_id}
```

The service automatically uses the verified full corpus when available and otherwise falls back to the portable demo corpus. Evidence enrichment upgrades independently as literature and graph artifacts become available.

## Planning pipeline

The current planner composes existing Neural Search infrastructure instead of inventing another model-only scoring layer:

1. **Data-form inference** identifies broad signal classes such as extracellular electrophysiology, optical imaging, EEG/MEG, MRI, intracellular recording, or behavior-only data.
2. **Required-signal checks** compare the dataset metadata with explicit requirements for the inferred data form.
3. **Analysis-family mapping** uses the reviewed method registry.
4. **Concrete method lookup** uses the method taxonomy to attach assumptions, limitations, and expected outputs.
5. **Target-dataset paper→method evidence** marks methods with known prior use on the target dataset.
6. **Related-dataset retrieval** finds nearby datasets and uses paper→method evidence on those neighbors as precedent candidates.
7. **Literature findings** enrich high-ranked methods when extracted findings are installed.
8. **Ranking** keeps feasibility, novelty state, and evidence separate.

## Feasibility is not novelty

A method receives one of these feasibility states:

- `supported_by_metadata`
- `conditional_missing_signals`
- `blocked_by_missing_signals`

It separately receives a novelty state:

- `existing_use_evidence`: current paper-method evidence already links this method to the target dataset;
- `possible_new_use_unverified`: no such evidence was found in the currently installed evidence graph.

`possible_new_use_unverified` is deliberately weak language. Missing evidence is not proof that an analysis has never been performed. Every candidate requires human review.

## Evidence tiers

Each candidate can carry several evidence records:

- `heuristic_candidate`: reviewed registry/taxonomy compatibility;
- `paper_linked_method_evidence`: method evidence already attached to a paper linked to the target dataset;
- `evidence_backed_bridge`: a related dataset has paper-method evidence for the candidate method;
- `extracted_literature_finding`: a relevant structured finding from the installed literature corpus.

These tiers are not interchangeable. A graph bridge should not be displayed as direct evidence that the target dataset supports the method.

## Required-signal blockers

The planner does not hide missing fields behind a scalar score. A candidate lists:

- all expected signal classes;
- signals that appear present;
- signals that are missing or not evidenced in metadata.

For example, a spike-train analysis candidate may require units/spike times/events. If trial events are not evidenced, the UI should say so even when the modality match is strong.

## Related-dataset precedent

Related datasets are discovered through the actual search engine. A precedent is only attached when the related dataset also has paper-method evidence for the candidate method.

This gives a useful research hypothesis:

> A method has been used on a dataset that appears experimentally related to this target dataset.

It does **not** establish that the target dataset is scientifically interchangeable with the precedent dataset. Species, acquisition, behavioral design, preprocessing, sample size, and other assumptions still require inspection.

## Evolution path

The next scientific validation layers should compare planner candidates against expert judgments:

1. Is the method technically feasible from the actual files, not just metadata?
2. Are the listed assumptions complete enough to prevent common misuse?
3. Does the evidence trail correctly distinguish direct use, related-dataset precedent, and general literature relevance?
4. Does the workbench identify useful analyses that researchers would not otherwise have considered?
5. Does a generated notebook execute successfully against the underlying dataset?

Those answers belong in the external-user benchmark and expert audit protocol, not in the method-priority score itself.
