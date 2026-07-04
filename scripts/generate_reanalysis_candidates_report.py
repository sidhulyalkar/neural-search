"""Generate reports/reanalysis_candidates_report.md.

Reports how many corpus datasets get a `dataset_old_dataset_new_method_candidate`
signal, broken down by data_form/analysis_family/technique, plus the honest
gap: which data_forms never produce a candidate because their analysis_family
has no entry in data/methods/method_registry.yaml.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from neural_search.awareness.taxonomy import DATA_FORMS
from neural_search.graph.method_registry_builder import load_method_registry
from neural_search.graph.reanalysis_candidates import build_reanalysis_candidate_edges

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "corpus"
    / "normalized"
    / "combined_corpus.jsonl"
    / "full_corpus_v09.jsonl"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "reanalysis_candidates_report.md"


def _load_corpus() -> list[dict]:
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    corpus = _load_corpus()
    edges = build_reanalysis_candidate_edges(corpus)

    datasets_with_candidates = {e.source_node_id for e in edges}
    family_counts = Counter(e.properties["analysis_family"] for e in edges)
    method_counts = Counter(e.target_node_id for e in edges)
    data_form_counts = Counter(e.properties["data_form"] for e in edges)
    has_papers_count = sum(1 for e in edges if e.properties["has_linked_papers"])

    registry = load_method_registry()
    linked_families = {link.analysis_family for link in registry.links}
    all_families_by_data_form = {
        form_id: set(form.analysis_families) for form_id, form in DATA_FORMS.items()
    }
    data_forms_with_no_candidate_family = sorted(
        form_id
        for form_id, families in all_families_by_data_form.items()
        if families and not (families & linked_families)
    )

    lines = [
        "# Reanalysis Candidates Report",
        "",
        f"- Corpus records scanned: {len(corpus)}",
        f"- Candidate edges (`dataset_old_dataset_new_method_candidate`): {len(edges)}",
        f"- Datasets with >=1 candidate: {len(datasets_with_candidates)}/{len(corpus)} "
        f"({100 * len(datasets_with_candidates) / len(corpus):.1f}%)",
        f"- Candidate edges on datasets with existing linked papers "
        f"(weak signal only, not a prior-usage proof): {has_papers_count}/{len(edges)}",
        "",
        "**Caveat:** every edge is a heuristic ('this dataset's profile matches a "
        "data form/analysis family this technique supports'), not a verified claim "
        "that the dataset hasn't already been analyzed this way. All edges carry "
        "`requires_human_review=True`. See the Methodology Registry work "
        "(`reports/methodology_coverage_report.md`) for why 10/27 analysis_families "
        "have no technique mapping yet.",
        "",
        "## Candidates by data form",
        "",
    ]
    for data_form_id, count in data_form_counts.most_common():
        lines.append(f"- {data_form_id}: {count}")

    lines.extend(["", "## Candidates by analysis family", ""])
    for family, count in family_counts.most_common():
        lines.append(f"- {family}: {count}")

    lines.extend(["", "## Candidates by technique", ""])
    for method_node_id, count in method_counts.most_common():
        lines.append(f"- {method_node_id}: {count}")

    lines.extend(
        [
            "",
            "## Data forms with zero candidate-eligible analysis families (open gap)",
            "",
        ]
    )
    if data_forms_with_no_candidate_family:
        lines.extend(f"- {form_id}" for form_id in data_forms_with_no_candidate_family)
    else:
        lines.append("- none")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
