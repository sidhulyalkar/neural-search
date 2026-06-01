# Task 27: Scientific Readiness Audit

Status: implemented for the v0.8 first slice.

This task adds a deterministic audit that summarizes corpus breadth, graph coverage, benchmark/calibration availability, and agent workflow readiness before ranking weights are promoted.

Implemented surfaces:

- `python -m neural_search.reports.scientific_readiness --out data/reports/readiness`
- `/api/corpus/readiness`
- JSON and Markdown report writers for automation and human review

The report intentionally warns when the corpus is small, species coverage is sparse, graph context is missing, paper links are weak, or benchmark/calibration reports were not supplied.
