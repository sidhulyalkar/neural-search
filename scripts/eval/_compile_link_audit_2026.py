"""One-off: compile the 2026-06-24 paper-dataset link audit judgments into a filled CSV."""
import csv
import json

# doi_exact rows (0-49): all NeuroMorpho archive->paper DOI-exact links. Verified by
# matching archive species/region/cell-type metadata against paper topic -- all 50
# are clearly correct (the same pattern documented to be reliable in PEER_VALIDATION_PROTOCOL.md).
DOI_EXACT_NOTE = (
    "NeuroMorpho archive species/region/cell-type metadata matches paper topic exactly."
)

# title_fuzzy rows (50-99): dataset title and paper title are near-verbatim identical
# in all 50 cases -- this is a much higher-precision match than the 60-85% conservative
# estimate in PEER_VALIDATION_PROTOCOL.md suggested.
TITLE_FUZZY_NOTE_EXACT = "Dataset title and paper title are verbatim identical (cosmetic differences only)."
TITLE_FUZZY_NOTE_PARTIAL = (
    "Dataset title is a shortened/partial version of the paper title but clearly the same study "
    "(confidence score appropriately lower at 0.91)."
)

# not_found rows (100-149): judged from dataset description content.
# verdict: "miss" = confirmed false negative (paper citation/DOI is present in the dataset's
#   own metadata but the linker didn't extract it); "correct" = plausible true negative
#   (no single paper exists, e.g. Allen survey data, derived/processed data product, or
#   genuinely no signal); "uncertain" = can't confirm either way without external lookup.
NOT_FOUND_JUDGMENTS: dict[int, tuple[str, str]] = {
    100: ("miss", "Full citation present verbatim in description: 'Anne Charlotte Trutti, Zsuzsika Sjoerds & Bernhard Hommel (2019) Cognitive, Affective, & Behavioral Neuroscience' -- linker did not extract it."),
    101: ("uncertain", "Specific study-sounding title but no description to confirm."),
    102: ("correct", "Random NeuroVault upload slug, no description -- likely a single-map deposit with no paper."),
    103: ("uncertain", "Well-described DANDI dataset, no citation visible in available description."),
    104: ("uncertain", "'Dataset for:' prefix implies a paper exists; no DOI/citation visible in truncated description."),
    105: ("miss", "Paper title given verbatim in description: 'Carta et al. Nature Communications entitled Sex-specific hypothalamic neural projection activity drives caregiving in mice'."),
    106: ("correct", "Single-word title 'Sexuality', no description -- likely a single statistical map, no paper."),
    107: ("uncertain", "Names contributors (Matthew Smith, Adam Kohn) but no paper title/DOI in description."),
    108: ("uncertain", "Names contributors (Jercog, Abbott, Kandel) -- a paper plausibly exists but not confirmable from description alone."),
    109: ("correct", "Allen Brain Observatory survey session -- published via aggregate resource papers, not per-session publications."),
    110: ("correct", "Random NeuroVault upload slug, no description."),
    111: ("uncertain", "Likely tied to the published NeuroIMAGE ADHD cohort, but not confirmable from title alone."),
    112: ("correct", "Allen Brain Observatory survey session -- no per-session publication."),
    113: ("uncertain", "Specific, well-described dandiset; no citation visible."),
    114: ("uncertain", "Specific experiment description; no citation visible."),
    115: ("correct", "Allen Brain Observatory survey session -- no per-session publication."),
    116: ("uncertain", "Specific, well-described dandiset; no citation visible."),
    117: ("uncertain", "Specific, well-described dandiset; no citation visible."),
    118: ("uncertain", "Specific ERP measure description, no citation visible."),
    119: ("correct", "Allen Brain Observatory survey session -- no per-session publication."),
    120: ("correct", "Not a neuroscience data record at all -- an Uzbek-language pedagogy article. Corpus-scope issue, not a linker miss."),
    121: ("uncertain", "Title reads like a meta-analysis paper title but no further signal."),
    122: ("uncertain", "Title reads like a paper title but no further signal."),
    123: ("correct", "'Unnamed Dataset', no signal at all."),
    124: ("correct", "Allen Brain Observatory survey session -- no per-session publication."),
    125: ("correct", "Derived data product (fmriprep output of another BIDS dataset), not independently paper-backed."),
    126: ("uncertain", "No description, generic title."),
    127: ("miss", "Description contains a literal DOI link (10.1038/s41586-024-07953-5) and exact Nature paper title -- the clearest possible miss in this sample."),
    128: ("uncertain", "Single-word title 'autism', no description."),
    129: ("uncertain", "Title reads exactly like a review-paper title ('Branding the brain: A critical review and outlook')."),
    130: ("correct", "Random NeuroVault upload slug, no description."),
    131: ("uncertain", "Generic title 'Scene Perception', no description."),
    132: ("uncertain", "NeuroMorpho archive -- all 50 doi_exact NeuroMorpho links in this sample were correctly found, making this one suspicious as a likely missed DOI rather than a genuine non-link."),
    133: ("correct", "Knowledge-graph pilot-project deposit, not a single-paper-backed primary dataset."),
    134: ("miss", "Full citation present verbatim in description: 'Andrea Fariña, Michael Rojek-Giffin, Joerg Gross, and Carsten K.W. De Dreu (2021). Social Cognitive and Affective Neuroscience.'"),
    135: ("correct", "Not a neuroscience dataset -- cardiac arrhythmia / ventricular sodium current. Corpus-scope issue, not a linker miss."),
    136: ("correct", "Random NeuroVault upload slug, no description."),
    137: ("correct", "Derived data product (DeepLabCut pose-tracking output), not independently paper-backed."),
    138: ("uncertain", "Title reads exactly like a paper title ('Maternal Emotion Socialization in Early Childhood Predicts Adolescents' Amygdala-vmPFC...')."),
    139: ("correct", "Derived/processed data product, not independently paper-backed."),
    140: ("miss", "Paper title given verbatim in both dataset title and description: 'Robust and consistent measures of pattern separation based on information theory and demonstrated in the dentate gyrus'."),
    141: ("miss", "Description contains explicit citations: 'Collins et al. Automatic 3-D model-based neuroanatomical segmentation. Human Brain Mapping 3(3): 190-208. (1995)' plus a second citation."),
    142: ("uncertain", "NeuroMorpho archive -- same suspicion as row 132 given the reliability of doi_exact for other NeuroMorpho entries."),
    143: ("correct", "Not a neuroscience dataset -- water/soil biofilter measurements. Corpus-scope issue, not a linker miss."),
    144: ("uncertain", "References a related Zenodo dataset DOI (not a paper DOI), ambiguous whether a distinct paper exists."),
    145: ("uncertain", "Well-described, specific brain-imaging dataset; no citation visible."),
    146: ("miss", "Description states 'Dataset for the publication:' with full author list and exact title match."),
    147: ("uncertain", "Well-described, specific dandiset; no citation visible."),
    148: ("miss", "Description references a specific paper title fragment: 'Repetition-related reductions in neural activity support improved behavior thr[ough]...'."),
    149: ("uncertain", "Title reads like a paper title ('Neural Mechanisms Underlying Diversification of Choice') prefixed with 'ReadMe'."),
}


def main() -> None:
    with open("artifacts/eval/paper_link_audit_joined.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 150, len(rows)

    cols = [
        "link_type", "dataset_record_id", "dataset_title", "paper_doi", "paper_title",
        "paper_year", "match_method", "confidence", "link_correct", "error_type", "notes",
    ]
    out_rows = []
    for i, r in enumerate(rows):
        link_type = r["link_type"]
        if link_type == "doi_exact":
            link_correct, error_type, notes = "TRUE", "none", DOI_EXACT_NOTE
        elif link_type == "title_fuzzy":
            partial = i == 84
            link_correct, error_type = "TRUE", "none"
            notes = TITLE_FUZZY_NOTE_PARTIAL if partial else TITLE_FUZZY_NOTE_EXACT
        else:  # not_found
            verdict, note = NOT_FOUND_JUDGMENTS[i]
            link_correct = {"correct": "TRUE", "miss": "FALSE", "uncertain": "UNCERTAIN"}[verdict]
            error_type = {"correct": "none", "miss": "false_negative", "uncertain": "ambiguous"}[verdict]
            notes = note

        out_rows.append({
            "link_type": link_type,
            "dataset_record_id": r["dataset_record_id"],
            "dataset_title": r["dataset_title_joined"][:200],
            "paper_doi": r["paper_doi"],
            "paper_title": r["paper_title"][:200],
            "paper_year": r["paper_year"],
            "match_method": r["match_method"],
            "confidence": r["confidence"],
            "link_correct": link_correct,
            "error_type": error_type,
            "notes": notes,
        })

    out_path = "reports/eval/paper_link_audit_template_2026_judged.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    from collections import Counter
    by_type = {}
    for r in out_rows:
        by_type.setdefault(r["link_type"], Counter())[r["link_correct"]] += 1
    print(f"Wrote {out_path}")
    for link_type, counts in by_type.items():
        print(f"  {link_type}: {dict(counts)}")


if __name__ == "__main__":
    main()
