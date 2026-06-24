"""One-off: compile the 2026-06-24 typed-field extraction audit judgments into a filled CSV."""
import csv
import json

# index = row order in reports/eval/typed_field_audit_template.csv (0-based, excluding header)
JUDGMENTS: dict[int, tuple[str, str, str]] = {
    0: ("TRUE", "", "spatial_frame=local appears to be a default rather than text-grounded, but not clearly wrong."),
    1: ("FALSE", "temporal_pattern", "Finding is about opposite-direction cell-fate outcomes (death vs differentiation), not a temporal/oscillatory pattern at all."),
    2: ("TRUE", "", ""),
    3: ("PARTIAL", "negation", "'not late' qualifies WHEN the (real, affirmative) prevention effect occurs, not negating the finding itself."),
    4: ("FALSE", "frequency_band", "'IL-1 beta' is a cytokine name; 'beta' regex matched the molecule, not the EEG/LFP beta band."),
    5: ("FALSE", "negation", "'inhibitory' describes the GABAergic input being blocked (the manipulation), not a negated result -- the finding itself is an affirmative increase in dopamine release."),
    6: ("PARTIAL", "temporal_pattern", "'oscillatory' inferred from the word 'rhythm' in a behavioral discrimination task, not an actual neural oscillation finding."),
    7: ("FALSE", "effect_scale", "'moderate to severe' tagged effect_scale=modest -- modest contradicts severe."),
    8: ("TRUE", "", ""),
    9: ("PARTIAL", "negation", "Explicit 'but not beta' construction is a real negation of the beta-band correlation specifically, missed by the negation detector (pattern set too narrow); frequency_band correctly extracts both bands but doesn't distinguish that beta was the non-significant one."),
    10: ("TRUE", "", ""),
    11: ("TRUE", "", ""),
    12: ("FALSE", "frequency_band", "'alpha-lipoic acid' is a chemical compound; 'alpha' regex matched the molecule name, not the alpha band."),
    13: ("TRUE", "", ""),
    14: ("TRUE", "", ""),
    15: ("FALSE", "negation", "'Suppression' is the finding's subject (posterior alpha oscillations were suppressed) -- an affirmative description of the phenomenon, not a negation of a claim."),
    16: ("TRUE", "", ""),
    17: ("TRUE", "", ""),
    18: ("FALSE", "negation", "'Baclofen blocked X' is the core affirmative pharmacological finding being reported, not a negated result."),
    19: ("FALSE", "negation", "'suppressed abnormal oscillatory activity' is the affirmative finding (the stimulation's effect), not a negation."),
    20: ("FALSE", "negation", "'Phasic ACh ... biphasic inhibitory followed by excitatory responses' describes a real physiological response, not a negated finding."),
    21: ("TRUE", "", ""),
    22: ("TRUE", "", ""),
    23: ("FALSE", "negation", "'cells inhibited during SPW' describes the observed cell population, not a negated finding -- the actual result ('fired rhythmically and phase-locked') is affirmative."),
    24: ("TRUE", "", "Correct use of negation for a genuine 'did not affect' construction -- contrasts with the inhibit/suppress/block false-positive pattern seen elsewhere in this sample."),
    25: ("TRUE", "", ""),
    26: ("PARTIAL", "temporal_pattern", "'oscillatory' applied to a multidien (multi-day-scale) rhythm -- defensible as 'periodic' but a different sense than the typical theta/gamma usage elsewhere in this field."),
    27: ("FALSE", "negation", "'inhibition' is a task/cognitive condition name (alongside 'change' and 'action activation'), not a description of suppressed neural activity -- the finding is an affirmative theta-power comparison across conditions."),
    28: ("FALSE", "frequency_band", "'A delta fibre' is a peripheral nerve fiber classification (A-beta/A-delta/C fibers), not the delta EEG/LFP band."),
    29: ("TRUE", "", ""),
    30: ("FALSE", "frequency_band", "'Amyloid-beta (A-beta)' is a protein, not the beta band -- same systematic collision as rows 4, 12, 28."),
    31: ("TRUE", "", "Correctly extracts both high_frequency and low_frequency from explicit text; negation correctly triggered by genuine 'absence of' construction."),
}


def main() -> None:
    with open("reports/eval/typed_field_audit_sample.jsonl", encoding="utf-8") as f:
        sample = [json.loads(line) for line in f]
    assert len(sample) == 32, len(sample)

    primary = ["negation", "frequency_band", "temporal_pattern", "spatial_frame"]
    context = ["condition", "effect_scale", "behavioral_measure", "population_type"]
    cols = ["finding_id", "paper_id", "finding_text", *primary, *context,
            "human_correct", "wrong_fields", "notes"]

    rows = []
    for i, rec in enumerate(sample):
        verdict, wrong_field, note = JUDGMENTS[i]
        row = {"finding_id": rec.get("finding_id", ""), "paper_id": rec.get("paper_id", ""),
               "finding_text": rec.get("finding_text", "")}
        for f in primary + context:
            v = rec.get(f)
            row[f] = "; ".join(v) if isinstance(v, list) else (v if v else "")
        row["human_correct"] = verdict
        row["wrong_fields"] = wrong_field
        row["notes"] = note
        rows.append(row)

    out_path = "reports/eval/typed_field_audit_template_2026_judged.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    verdicts = Counter(r["human_correct"] for r in rows)
    wrong = Counter(r["wrong_fields"] for r in rows if r["wrong_fields"])
    n_true, n_partial = verdicts["TRUE"], verdicts["PARTIAL"]
    print(f"Wrote {out_path}")
    print("Verdicts:", dict(verdicts))
    print("Wrong-field breakdown:", dict(wrong))
    print(f"Strict precision: {n_true}/{len(rows)} = {n_true/len(rows)*100:.1f}%")
    print(f"Weighted precision: {(n_true + 0.5*n_partial)/len(rows)*100:.1f}%")


if __name__ == "__main__":
    main()
