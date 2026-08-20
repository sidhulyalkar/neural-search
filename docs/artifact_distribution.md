# Versioned artifact distribution and lineage

Neural Search keeps code, small fixtures, and canonical evaluation inputs in Git while large research assets live outside Git. The artifact subsystem turns those external assets into reproducible, checksum-verified inputs instead of implicit workstation state.

## The contract

A published release is immutable at **two levels**.

The release index pins:

- bundle name and version;
- immutable HTTPS manifest URL;
- SHA-256 of the manifest itself;
- compatibility group such as `corpus:v09`.

The pinned bundle manifest then records:

- each artifact ID and cache-relative destination;
- HTTPS source URL;
- exact byte size and SHA-256 digest;
- artifact version and content-addressed lineage ID;
- parent lineage IDs for derived artifacts;
- optional source commit and metadata.

Changing either the manifest bytes or any artifact bytes requires a new release version. A stable `name@version` must never be edited in place.

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

Each installed artifact also records the size and modification time observed during SHA verification. Normal runtime resolution uses that inexpensive checkpoint. If the file changes afterward, Neural Search refuses to resolve it from the bundle lock until it is re-verified.

## Fetch and verify

List published releases:

```bash
neural-search artifacts releases
```

Once a real release has been published, install it with its immutable ref, for example:

```bash
neural-search artifacts fetch neural-search-researcher@0.9.2
```

The example above is illustrative until that ref actually appears in `artifacts releases`.

For normal release-index fetching, Neural Search verifies the **manifest SHA-256 first**, then verifies every artifact declared by that manifest.

The downloader:

1. accepts HTTPS remote sources by default;
2. rejects unsafe `..` or absolute artifact paths;
3. enforces the manifest's declared byte size;
4. streams the file into a temporary path;
5. verifies SHA-256 before installation;
6. atomically replaces the destination;
7. writes a validated lineage sidecar;
8. records a verified stat checkpoint in the local artifact lock;
9. removes stale pins from an older version of the same bundle.

Re-hash every pinned file explicitly:

```bash
neural-search artifacts verify
```

Inspect pins without a full multi-gigabyte re-hash:

```bash
neural-search artifacts lock
```

The `/system` runtime page also uses the lightweight verified stat checkpoint. It does not claim to have re-hashed all artifact bytes on every page load.

Local-file manifest/artifact sources exist only for controlled tests or institutional mirrors and must be explicitly enabled with `--allow-local-files`. Passing `--manifest` directly is a testing/mirror escape hatch: artifact bytes are still pinned by that manifest, but the manifest is not considered an indexed immutable release unless it is also added to the release index with its own digest.

## Content-addressed lineage

Generated artifacts receive sidecars:

```text
file.jsonl.neural-search.json
```

Directories use:

```text
<directory>/.neural-search-artifact.json
```

A lineage sidecar records the artifact's content digest plus exact parent lineage IDs. The lineage ID is itself deterministically derived from artifact ID, artifact version, and content SHA-256. Malformed or internally inconsistent lineage records are rejected.

For example:

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

If the artifact bytes change after stamping, bundle publication fails. Rebuild/review the output and stamp a new lineage rather than reusing stale provenance.

## Compatibility semantics

Neural Search distinguishes four situations:

1. **compatible**: installed parent lineage matches the child's declared parent lineage;
2. **unknown / parent not local**: the child records a parent lineage, but the source bytes are intentionally not installed;
3. **unknown / untracked**: an old local artifact exists without a lineage sidecar;
4. **incompatible**: an installed parent has a different lineage, an artifact version/group conflicts, or a required parent relationship was omitted.

The second case is important for distribution. A researcher should not need all raw archive downloads merely to use a verified normalized-corpus release. The lineage reference remains auditable even when the parent bytes are remote.

## Publish a bundle

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

The command refuses unusable artifacts, directory artifacts that have not been archived, unstamped artifacts by default, changed bytes with stale lineage, missing required parent declarations, version conflicts, and compatibility-group conflicts.

Upload the manifest itself to an immutable HTTPS location. Then pin it in the repository release index:

```bash
neural-search artifacts release-add \
  --manifest releases/neural-search-researcher-0.9.2.json \
  --manifest-url https://example.org/neural-search/manifests/neural-search-researcher-0.9.2.json
```

`release-add` computes the manifest SHA-256, writes release-index schema v2, and refuses to mutate an existing `name@version`. If a release must change, increment its version.

Commit the updated `data/artifacts/releases/index.json` through a normal reviewed PR.

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
