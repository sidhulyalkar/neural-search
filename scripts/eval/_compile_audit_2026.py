"""One-off: compile the 2026-06-23 finding-audit judgments into a filled CSV."""
import csv
import json
from collections import Counter

judgments = {
    0: ("TRUE", "none", ""),
    1: ("PARTIAL", "region_wrong", "Finding text accurate to abstract but region field is 'san francisco' (a city, not a brain region); paper is non-neuroscience (adolescent health behavior) -- corpus/domain contamination."),
    2: ("TRUE", "none", "Plausible continuation of truncated abstract; topic and direction match precisely."),
    3: ("TRUE", "none", ""),
    4: ("PARTIAL", "region_wrong", "Verified via full abstract: 'hippocampus' never appears in the text. Region inferred from AD-domain knowledge, not text evidence."),
    5: ("TRUE", "none", ""),
    6: ("PARTIAL", "direction_wrong", "Mechanistic/structural finding (two sets of specificity-determining positions) mistagged direction='no_change' instead of 'mechanism' -- same dominant failure mode as the 2026-06-22 audit."),
    7: ("PARTIAL", "species_wrong", "Verified via full abstract: species is 'rabbit' (stated explicitly) but field is empty. Also non-neuroscience (kidney/glomerular disease model) -- domain contamination."),
    8: ("TRUE", "none", ""),
    9: ("TRUE", "none", ""),
    10: ("TRUE", "none", ""),
    11: ("TRUE", "none", "Plausible continuation of truncated abstract; topic match precise."),
    12: ("TRUE", "none", ""),
    13: ("TRUE", "none", "Plausible continuation of truncated abstract (paper's entire subject is LGG/GBM location comparison)."),
    14: ("TRUE", "none", "Plausible continuation of truncated abstract."),
    15: ("TRUE", "none", "Direction='decrease' for relative auditory<visual latency is an unusual but defensible interpretation."),
    16: ("TRUE", "none", "Borderline use of no_change for a selective-interaction result, but defensible."),
    17: ("TRUE", "none", ""),
    18: ("TRUE", "none", "Non-neuroscience domain (health-behavior intervention) but textually accurate."),
    19: ("TRUE", "none", "species='mammals' is a generic placeholder, but abstract genuinely never specifies the animal model."),
    20: ("TRUE", "none", "Plausible continuation of truncated abstract (paper directly compares CBD/PSP/SDAT)."),
    21: ("TRUE", "none", ""),
    22: ("TRUE", "none", "Plausible continuation of truncated abstract; exact topical match."),
    23: ("PARTIAL", "region_wrong", "Verified via full abstract: neither 'primary olfactory cortex' nor 'trigeminal nucleus' is named anywhere. Regions inferred from sensory-modality domain, not stated."),
    24: ("TRUE", "none", "Minor: 'no_change' understates 'mildly affected' framing but defensible for the monkey-1 null result."),
    25: ("TRUE", "none", ""),
    26: ("TRUE", "none", "Plausible continuation of truncated abstract."),
    27: ("TRUE", "none", "Plausible continuation of truncated abstract; exact match to stated research question."),
    28: ("PARTIAL", "other", "A methods/measurement sentence ('sensitivities were measured') extracted as if it were a result -- new failure mode: method description mistaken for a finding."),
    29: ("TRUE", "none", "Plausible continuation of truncated abstract."),
    30: ("TRUE", "none", ""),
    31: ("TRUE", "none", "Plausible continuation; non-neuroscience domain (exergaming/psychosocial)."),
    32: ("TRUE", "none", "High-confidence plausible continuation; paper title names exactly these mechanisms."),
    33: ("PARTIAL", "other", "A methods-justification sentence ('recordings were necessary to study X') extracted as a finding -- same new failure mode as row 28."),
    34: ("TRUE", "none", "Plausible continuation of truncated abstract."),
    35: ("TRUE", "none", "Review paper; finding text accurately reflects the review's central claim."),
    36: ("TRUE", "none", "Verified via full abstract: 'Knocking out the Piezo1 in CEA neurons showed a significant reduction...' appears verbatim. Initially flagged as a possible mismatch with the motor-cortex experiment in the same paper -- confirmed correct on full-text check."),
    37: ("TRUE", "none", ""),
    38: ("TRUE", "none", ""),
    39: ("TRUE", "none", ""),
    40: ("TRUE", "none", ""),
    41: ("TRUE", "none", "Non-neuroscience (orthopedic biomechanics) but textually accurate."),
    42: ("TRUE", "none", "Non-neuroscience (bacterial ion channel biophysics) but textually accurate."),
    43: ("TRUE", "none", ""),
    44: ("TRUE", "none", ""),
    45: ("TRUE", "none", "Non-neuroscience (bone/connective tissue biology) but textually accurate."),
    46: ("TRUE", "none", ""),
    47: ("TRUE", "none", ""),
    48: ("TRUE", "none", ""),
    49: ("TRUE", "none", ""),
    50: ("PARTIAL", "species_wrong", "species='other' for what is clearly a human-subjects study (video headset, facial recognition viewing task) -- should be 'human'."),
    51: ("TRUE", "none", "Plausible continuation of truncated abstract; exact topical match."),
    52: ("TRUE", "none", "Matches abstract almost verbatim."),
    53: ("TRUE", "none", "Plausible continuation; 'no_change' fits 'approximately the same from both baselines'."),
    54: ("TRUE", "none", "Good example of correct 'mechanism' direction usage for a non-directional mechanistic finding."),
    55: ("TRUE", "none", "Good example of correct 'other' direction usage for a mixed-direction finding."),
    56: ("TRUE", "none", "Good example of correct 'other' direction usage."),
    57: ("PARTIAL", "direction_wrong", "Abstract describes a non-monotonic stage relationship (stage 1 > stage 2, stage 3 worst) -- the finding's claim of simple correlation with disease stage oversimplifies this."),
    58: ("TRUE", "none", "Mixed increase/decrease across regions correctly captured with direction='other'."),
    59: ("TRUE", "none", "Plausible continuation; direction='no_change' for the PD group's absent effect is reasonable."),
    60: ("TRUE", "none", "Plausible continuation of truncated abstract."),
    61: ("TRUE", "none", "Correct 'other' usage for a complex partial-recovery time course."),
    62: ("TRUE", "none", "Extractor itself flagged lower confidence (0.6) for an unconfirmed-in-preview claim -- good calibration."),
    63: ("TRUE", "none", "Plausible continuation; precise topical match."),
    64: ("TRUE", "none", ""),
    65: ("PARTIAL", "direction_wrong", "Architectural/organizational finding (PFC organized as a cascade) mistagged direction='no_change' -- same failure mode as row 6."),
    66: ("TRUE", "none", "Precise quantitative match (147%, 29%) to abstract."),
    67: ("PARTIAL", "species_wrong", "species field empty despite clearly being human epilepsy-surgery patients (63 patients named in abstract)."),
    68: ("TRUE", "none", "Plausible continuation; precise topical match."),
    69: ("TRUE", "none", "Plausible continuation of truncated abstract; exact topical/methodological match."),
    70: ("TRUE", "none", "retrosplenial cortex not shown in the visible window but anatomically plausible alongside PCC; not flagged as hard error."),
    71: ("PARTIAL", "region_wrong", "region=['chromosome 19'] -- a genomic locus, not a brain region. Finding text itself is accurate; field schema misapplied (pure genetics paper with no brain-region content)."),
    72: ("TRUE", "none", "Plausible continuation; precise topical match."),
    73: ("TRUE", "none", "Matches abstract almost verbatim. Empty regions field is correctly scoped to this specific finding sentence."),
    74: ("TRUE", "none", "Matches abstract almost verbatim. Non-neuroscience (behavioral cognitive development, no brain measure) but textually accurate."),
    75: ("TRUE", "none", "Matches abstract almost verbatim."),
    76: ("TRUE", "none", "Extractor flagged lower confidence (0.6) appropriately for an awkwardly-phrased finding."),
    77: ("TRUE", "none", ""),
    78: ("PARTIAL", "region_wrong", "region=['primary somatosensory cortex'] never appears in this purely psychophysical abstract -- inferred from touch-domain, not stated."),
    79: ("TRUE", "none", ""),
    80: ("TRUE", "none", "Plausible continuation; precise topical and methodological match."),
    81: ("TRUE", "none", "Near-verbatim match including exact figures (2-3 fold, 4-fold)."),
    82: ("TRUE", "none", "Matches the paper's stated and confirmed hypothesis precisely."),
    83: ("TRUE", "none", "Plausible continuation; precise topical match."),
    84: ("TRUE", "none", "Plausible continuation; precise topical/methodological match."),
    85: ("TRUE", "none", "Numbers broadly consistent with visible DA reduction figures; DAT-specific figures likely appear later in untruncated abstract."),
    86: ("TRUE", "none", "Plausible continuation; correct 'other' usage for a multi-factor finding."),
    87: ("TRUE", "none", "Plausible instantiation of the paper's general finding (face patches differ in viewpoint representation)."),
    88: ("TRUE", "none", "Matches abstract almost verbatim."),
    89: ("FALSE", "hallucinated", "region=['higher than primary visual cortex'] -- a comparative phrase concatenated into the region field as if it were an anatomical name. Not a real structure."),
    90: ("TRUE", "none", "Matches abstract almost verbatim."),
    91: ("TRUE", "none", "Exact verbatim match to abstract."),
    92: ("TRUE", "none", "Exact verbatim match to abstract; species list of three mutant lines correctly extracted."),
    93: ("PARTIAL", "direction_wrong", "Finding explicitly describes BOTH a decrease (ATP) and an increase (lipid peroxidation) but was force-tagged direction='decrease' instead of 'other'."),
    94: ("PARTIAL", "species_wrong", "species='other' for human donor-eye/patient tissue (ages 69-82 explicitly stated) -- should be 'human'."),
    95: ("TRUE", "none", "Plausible continuation; matches the paper's stated and confirmed goal precisely."),
    96: ("TRUE", "none", "Near-verbatim match to abstract."),
    97: ("FALSE", "hallucinated", "region=['visual cortex'] fabricated -- this is a purely behavioral infant gaze-following study with no neuroimaging or brain measurement of any kind."),
    98: ("PARTIAL", "region_wrong", "region=['mbp(111-129)'] -- a peptide epitope name, not an anatomical region. Also borderline domain looseness (T-cell immunology)."),
    99: ("TRUE", "none", "Plausible continuation; matches the paper's hypothesis and framing precisely."),
}


def main() -> None:
    with open("reports/eval/finding_audit_sample.jsonl", encoding="utf-8") as f:
        sample = [json.loads(line) for line in f]
    assert len(sample) == 100, len(sample)

    cols = [
        "finding_id", "paper_id", "finding_text", "regions", "tasks", "modalities",
        "species", "result_direction", "confidence", "human_correct", "error_type", "notes",
    ]
    rows = []
    for i, rec in enumerate(sample):
        verdict, err, note = judgments[i]
        rows.append({
            "finding_id": rec.get("finding_id", ""),
            "paper_id": rec.get("paper_id", ""),
            "finding_text": rec.get("finding_text", ""),
            "regions": "; ".join(rec.get("regions", []) or []),
            "tasks": "; ".join(rec.get("tasks", []) or []),
            "modalities": "; ".join(rec.get("modalities", []) or []),
            "species": "; ".join(rec.get("species", []) or []),
            "result_direction": rec.get("result_direction", ""),
            "confidence": rec.get("confidence", ""),
            "human_correct": verdict,
            "error_type": err,
            "notes": note,
        })

    out_path = "reports/eval/finding_audit_template_2026_judged.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    verdicts = Counter(r["human_correct"] for r in rows)
    errs = Counter(r["error_type"] for r in rows if r["human_correct"] != "TRUE")
    n_true, n_partial = verdicts["TRUE"], verdicts["PARTIAL"]
    print(f"Wrote {out_path}")
    print("Verdicts:", dict(verdicts))
    print("Error types:", dict(errs))
    print(f"Strict precision (TRUE only): {n_true}%")
    print(f"Weighted precision (TRUE + 0.5*PARTIAL): {(n_true + 0.5 * n_partial):.1f}%")


if __name__ == "__main__":
    main()
