# Manuscript TODO List

This document tracks remaining work for the Neural Search manuscript.

## Completed (This Revision)

- [x] Soften "state-of-the-art" claims in abstract and contributions
- [x] Add Claim Status and Evidence table (Table 1)
- [x] Add compact graph schema tables (Tables 2, 3)
- [x] Add analysis affordance requirements table (Table 4)
- [x] Expand related work: PROV-O, RO-Crate, LinkML, EBRAINS, openMINDS
- [x] Expand benchmark methodology section
- [x] Add error taxonomy for failure analysis
- [x] Expand limitations section with specific weaknesses
- [x] Add future experiments section with validation roadmap
- [x] Fix LaTeX float specifiers ([h] -> [t])
- [x] Add tabularx package for table flexibility
- [x] Add reproducibility statement with script paths

## Pending (Before Submission)

### Content

- [ ] Complete bibliography (some entries need DOIs/URLs)
- [ ] Add Figure 5: Example search result screenshot or diagram
- [ ] Add Table 6: Error taxonomy with example failures
- [ ] Proofread all sections for clarity and consistency
- [ ] Verify all metric claims match latest benchmark run

### Experiments

- [ ] Run full baseline ladder and update Table 5 if needed
- [ ] Run adversarial hard-negative benchmark
- [ ] Generate violation report for Appendix
- [ ] Measure statistical significance (paired t-test or bootstrap)

### Formatting

- [ ] Check all figure captions are informative
- [ ] Verify all tables fit within column width
- [ ] Check for any remaining overfull hbox warnings
- [ ] Verify cross-references are correct

## Future Revisions

### After Initial Feedback

- [ ] Address reviewer concerns about corpus size
- [ ] Add inter-annotator agreement metrics
- [ ] Consider adding user study results
- [ ] Expand to additional repositories (Allen, NeMO)

### For Camera-Ready

- [ ] High-resolution figures
- [ ] Acknowledgments section
- [ ] Funding statement
- [ ] Author contributions

## Section-by-Section Status

| Section | Status | Priority | Notes |
|---------|--------|----------|-------|
| Abstract | DONE | - | Softened claims |
| Introduction | DONE | - | Claim status table added |
| Related Work | DONE | - | Expanded coverage |
| Formalization | DONE | - | No changes needed |
| Knowledge Graph | DONE | - | Schema tables added |
| Affordances | DONE | - | Requirements table added |
| Retrieval | DONE | - | No changes needed |
| Embeddings | DONE | - | No changes needed |
| Experiments | REVISED | HIGH | Protocol expanded |
| Discussion | REVISED | - | Limitations expanded |
| Future Work | REVISED | - | Experiments roadmap added |
| Conclusion | DONE | - | No changes needed |
| References | PARTIAL | MED | Some entries incomplete |
| Appendix | TODO | LOW | Consider adding |

## Notes for Codex

When resuming work:

1. Run `python -m neural_search.evaluation.run_benchmark --suite demo_v02` to verify metrics
2. Check `docs/paper/CLAIM_STATUS_AND_EVIDENCE.md` for claim verification
3. See `docs/paper/EXPERIMENT_ROADMAP.md` for implementation priorities
4. Phase 4-9 implementation should precede final manuscript updates

## Questions for Authors

1. Target venue: ICLR, NeurIPS, Neuroinformatics, or other?
2. Acceptable corpus size for initial submission?
3. Priority: broader coverage or deeper validation?
4. Include user study in initial submission or defer?

---

*Last updated: 2026-05-26*
