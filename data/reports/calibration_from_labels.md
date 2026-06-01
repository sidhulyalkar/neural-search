# Calibration Analysis Report

**Suite:** from_labels

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Samples | 30 |
| Positive Samples | 11 |
| Negative Samples | 19 |
| **ECE** | 0.1791 |
| **MCE** | 0.4962 |
| **Brier Score** | 0.1641 |
| Mean Confidence | 0.3094 |
| Mean Accuracy | 0.3667 |
| Calibration Slope | 1.4047 |

## Interpretation

- Moderate calibration (ECE < 0.20)
- System is underconfident (slope > 1.1)

## Over/Under Confidence

- Overconfidence Rate: 60.0%
- Underconfidence Rate: 33.3%

## Reliability Diagram

| Bin Range | Count | Confidence | Accuracy | Error |
|-----------|-------|------------|----------|-------|
| [0.0, 0.1) | 4 | 0.035 | 0.000 | 0.035 |
| [0.1, 0.2) | 3 | 0.141 | 0.333 | 0.193 |
| [0.2, 0.3) | 7 | 0.241 | 0.000 | 0.241 |
| [0.3, 0.4) | 9 | 0.345 | 0.444 | 0.099 |
| [0.4, 0.5) | 4 | 0.426 | 0.750 | 0.324 |
| [0.5, 0.6) | 1 | 0.504 | 1.000 | 0.496 |
| [0.7, 0.8) | 1 | 0.719 | 1.000 | 0.281 |
| [0.9, 1.0) | 1 | 1.000 | 1.000 | 0.000 |

## Recommendations

- Consider temperature scaling to reduce overconfidence
- Focus calibration efforts on confidence ranges: [0.5, 0.6), [0.4, 0.5)

## Calibration Quality Assessment

Overall Quality: ⚠️ **Fair** (ECE < 0.20)

⚠️ System tends to be **underconfident** (low scores for relevant results).