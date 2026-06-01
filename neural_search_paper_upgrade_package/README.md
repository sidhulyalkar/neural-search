# Neural Search Paper Upgrade Kit

This package contains a structured set of instructions for making the Neural Search whitepaper more reputable, more submission-ready, and more experimentally grounded.

## Files

1. `01_paper_expansion_instructions.md`  
   Detailed instructions for upgrading the manuscript from a visionary whitepaper into a credible systems/retrieval research paper.

2. `02_experiment_and_validation_roadmap.md`  
   A staged experiment plan covering baseline ladders, hard-negative benchmarks, affordance validation, graph/link-prediction experiments, latent neural signatures, and robustness testing.

3. `03_latex_overleaf_fixes.md`  
   Specific fixes for the Overleaf warnings you saw, including float placement, overfull boxes, underfull boxes, long math, and long `texttt` strings.

4. `04_claude_paper_revision_prompt.md`  
   A copy-paste prompt for Claude to revise the paper prose, structure, claims, related work, benchmark section, and future experiments.

5. `05_codex_experiment_implementation_prompt.md`  
   A copy-paste prompt for Codex to implement the evaluation and validation infrastructure feature-by-feature.

6. `06_master_instruction_spec.xml`  
   A repo-friendly XML instruction set containing paper, experiment, Claude, and Codex specifications.

7. `07_experiments_manifest.yaml`  
   A machine-readable experiment manifest that can be used to scaffold configs, issues, or task files.

8. `08_related_work_sources_to_verify.md`  
   A source checklist for related work and standards. Verify all citation details before adding to the final bibliography.

9. `09_reputation_checklist.md`  
   A compact checklist for determining whether the paper is becoming more credible, reproducible, and defensible.

## Core thesis to preserve

Neural Search is not only a search engine for neuroscience datasets. It is a retrieval framework for reusable experimental contexts, where datasets are matched by typed scientific meaning, graph relationships, provenance, and analysis affordances.

## Important principle

Do not invent new benchmark numbers. Use placeholders such as `TODO`, `to be evaluated`, or `not yet measured` when results have not been run.
