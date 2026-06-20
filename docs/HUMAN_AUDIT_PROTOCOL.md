# Human Audit Protocol

## When to Audit

The `artifacts/eval/audit_queue.jsonl` contains pairs where:
- LFs strongly disagreed (variance > 0.5), OR
- A hard-negative pattern was detected but not confirmed

These are the highest-priority pairs for human review.

## Workflow

### 1. Generate the audit queue

```bash
python scripts/obsidian/export_audit_queue.py \
    --audit-queue artifacts/eval/audit_queue.jsonl \
    --vault obsidian_vault
```

This creates one Markdown note per candidate in
`obsidian_vault/05_Annotations/Human Audits/`.

### 2. Open Obsidian

Open `obsidian_vault/` as a vault. Each note shows:
- The query text and scientific goal
- The hard negative constraints for that query
- The dataset's metadata (title, modalities, species, regions, tasks)
- The individual LF votes and their confidence
- A Human Audit Checklist section

### 3. Label each note

Edit the YAML frontmatter at the top of the note:

```yaml
label: 2
confidence: 0.9
audit_status: done
```

Then check off items in the Human Audit Checklist.

**Write-safety guarantee:** these three fields will never be overwritten
by automated re-export scripts. See `neural_search/obsidian/io.py`.

### 4. Import completed audits to gold qrels

```bash
python scripts/obsidian/import_audits.py \
    --vault obsidian_vault \
    --out artifacts/qrels_gold.jsonl
```

Only notes with `audit_status: done` are imported.

## Label Definitions

| Label | Meaning |
|---|---|
| 3 | All required properties present; directly supports the query intent |
| 2 | Most required properties present; partially supports the query |
| 1 | Tangentially related; would need adaptation or supplementation |
| 0 | Not relevant — or a hard negative (superficially relevant but wrong) |

## Confidence Guidance

| Confidence | When to use |
|---|---|
| 1.0 | Completely certain (e.g., obviously irrelevant, or exact match) |
| 0.9 | High confidence with minor uncertainty |
| 0.7 | Moderate certainty — some ambiguity in the query or metadata |
| 0.5 | Significant uncertainty — recommend flagging for second review |

## Which Metrics Use Gold

Only `artifacts/qrels_gold.jsonl` may be cited in the whitepaper or
scientific claims. Silver/bronze qrels are for development iteration only.

Running metrics with a non-gold qrels file emits a stderr warning:

```
WARNING: Using SILVER qrels. Results from silver labels should NOT be
cited as scientific validation. Use gold qrels for whitepaper claims.
```

## Audit Priority Order

The audit queue is sorted by `audit_priority` descending. High-priority
entries are:
1. Pairs where `lf_hard_negative` triggered (critical to verify)
2. Pairs with high label variance across LFs (disagreement)
3. Pairs from sources with low coverage in the gold set

## Dataview Dashboard

In Obsidian, create `obsidian_vault/08_Dashboards/Audit Progress.md`:

```dataview
TABLE label, confidence, audit_status
FROM "05_Annotations/Human Audits"
WHERE audit_status != "done"
SORT file.mtime DESC
```
