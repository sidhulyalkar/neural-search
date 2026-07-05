# Claim Coverage Dashboard

## Active Claims by Direction

```dataview
TABLE direction, regions, consensus_confidence, n_supporting_findings, status
FROM "10_Claims"
WHERE type = "claim" AND status = "active"
SORT consensus_confidence DESC
LIMIT 50
```

## Contested Claims

```dataview
TABLE statement, direction, regions, n_contradicting_findings
FROM "10_Claims"
WHERE status = "contested"
SORT n_contradicting_findings DESC
```

## Paper Coverage

```dataview
TABLE title, n_findings, linked_datasets
FROM "09_Literature"
WHERE type = "paper"
SORT n_findings DESC
LIMIT 30
```
