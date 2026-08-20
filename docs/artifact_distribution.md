# Versioned artifact distribution and lineage

Neural Search keeps code, small fixtures, and canonical evaluation inputs in Git while large research assets live outside Git. The artifact subsystem turns those external assets into reproducible, checksum-verified inputs instead of implicit workstation state.

## The contract

A published bundle is immutable. Its manifest records:

- bundle name and version;
- a compatibility group such as `corpus:v09`;
- each artifact ID and runtime destination;
- HTTPS source URL;
- exact byte size and SHA-256 digest;
- artifact version and content-addressed lineage ID;
- parent lineage IDs for derived artifacts;
- optional source commit and metadata.

Changing any bytes requires a new artifact digest and therefore a new lineage ID. Published manifests and object URLs must not be edited in place.

The repository ships `data/artifacts/releases/index.json` as the release-index contract. It is intentionally empty until actual immutable objects are published. Do not add fake URLs to make the UI look complete.

## Local cache and lock

By default, downloaded artifacts are stored under:

```text
~/.cache/neural-search/artifacts/<bundle>/<version>/...
```

The active pin set is recorded in:

```text
.neural-search/artifact-lock.json
```

Both locations can be overridden with `NEURAL_SEARCH_ARTIFACT_CACHE` and `NEURAL_SEARCH_ARTIFACT_LOCK`.

The lock is local state and is ignored by Git. Reproducibility manifests embed a snapshot of the lock so experiment outputs can record what was installed without committing the workstation cache.

## Fetch and verify

List published releases:

```bash
neural-search artifacts releases
```

Install one immutable bundle:

```bash
neural-search artifacts fetch neural-search-researcher@0.9.2
```

The downloader:

1. accepts HTTPS remote sources by default;
2. rejects unsafe `..` or absolute artifact paths;
3. enforces the manifest's declared byte size;
4. streams the file into a temporary path;
5. verifies SHA-256 before installation;
6. atomically replaces the destination;
7. writes the lineage sidecar;
8. updates the local artifact lock.

Re-verify every pinned file later:

```bash
neural-search artifacts verify
```

Inspect pins:

```bash
neural-search artifacts lock
```

Local-file sources exist only for controlled tests or institutional mirrors and must be explicitly enabled for artifact bytes with `--allow-local-files`.

## Content-addressed lineage

Generated artifacts receive sidecars:

```text
file.jsonl.neural-search.json
```

Directories use:

```text
<directory>/.neural-search-artifact.json
```

A lineage sidecar records the artifact's content digest plus exact parent lineage IDs. For example:

```text
raw source payloads
  -> full corpus
      -> dense embeddings
      -> knowledge graph
      -> coverage ledger
      -> reanalysis report
          -> evaluation/reporting outputs
```

A filename like `v09` is not provenance. The lineage ID is.

### Stamp a generated artifact

Parents must be stamped first so derivations are explicit:

```bash
neural-search artifacts stamp raw_corpus_inputs --version 2026-08-19
neural-search artifacts stamp full_corpus_v09 --version 0.9
neural-search artifacts stamp production_graph --version 0.9
neural-search artifacts stamp dense_field_embeddings --version 0.9
```

Portable committed fixtures usually do not need sidecars because Git already provides immutable identity.

## Compatibility semantics

Neural Search distinguishes four situations:

1. **compatible**: installed parent lineage matches the child's declared parent lineage;
2. **unknown / parent not local**: the child records a parent lineage, but the source bytes are intentionally not installed;
3. **unknown / untracked**: an old local artifact exists without a lineage sidecar;
4. **incompatible**: an installed parent has a different lineage, an artifact version/group conflicts, or a required parent relationship was omitted.

The second case is important for distribution. A researcher should not need all raw archive downloads merely to use a verified normalized-corpus release. The lineage reference remains auditable even when the parent bytes are remote.

## Publish a bundle manifest

First upload immutable artifact bytes to the chosen HTTPS origin. Then build a manifest from locally stamped artifacts:

```bash
neural-search artifacts bundle-build \
  neural-search-researcher 0.9.2 \
  full_corpus_v09 production_graph dense_field_embeddings coverage_ledger \
  --compatibility-group corpus:v09 \
  --source-base-url https://example.org/neural-search/0.9.2 \
  --source-commit <git-sha> \
  --output releases/neural-search-researcher-0.9.2.json
```

The command refuses unusable artifacts, directory artifacts that have not been archived, unstamped artifacts by default, missing required parent declarations, and compatibility-group conflicts.

Once the manifest itself is hosted immutably, add its URL to `data/artifacts/releases/index.json` in a normal reviewed PR.

## Reproducibility manifests

```bash
neural-search profile manifest researcher \
  --output reports/reproducibility/researcher.json
```

Schema v2 includes:

- Git commit;
- Python/platform state;
- profile readiness and health;
- capability availability;
- compatibility results;
- local artifact lock snapshot;
- artifact lineages or checksums;
- declared operating commands and resource notes.

This makes corpus, graph, embedding, and evaluation state part of the scientific record rather than an invisible prerequisite.
