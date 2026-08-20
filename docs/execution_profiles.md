# Execution Profiles

Neural Search is intentionally more than one runtime. A first-time user should not need the same dependencies, data assets, or hardware as a maintainer rebuilding dense embeddings and the production graph.

The supported profiles make those operating modes explicit and machine-checkable.

## Choose the smallest profile that does your job

| Profile | Best for | Install | Required state | Typical resources |
| --- | --- | --- | --- | --- |
| `demo` | First-time users, teaching, product evaluation, contributors | `make setup` | Committed ontology + demo fixtures | CPU, no external services |
| `researcher` | Searching a real local corpus, inspecting evidence, generating reuse outputs | `make install-researcher` | Current normalized corpus | CPU supported; GPU useful for rebuilding dense embeddings |
| `corpus-builder` | Refreshing source data, normalization, enrichment, embeddings, graph construction | `make install-corpus-builder` | Local raw source payloads | Network access; GPU recommended for dense builds |
| `evaluator` | Retrieval benchmarks, qrels analysis, scientific regression checks | `make install-evaluator` | Committed canonical queries + qrels | CPU for portable baseline; local graph/dense assets for full ablations |
| `full-stack` | Maintainers operating the complete product and infrastructure | `make install-full-stack` | Real corpus + graph + dense embeddings | Postgres/Redis capable; GPU recommended for rebuilds |

The profile names describe capability, not scientific evidence quality. Running `full-stack` does not make a result more scientifically valid than a `demo` result. Evidence tier, benchmark provenance, corpus state, and the actual analysis determine that.

## Inspect a profile before running it

```bash
neural-search profile list
neural-search profile show researcher
neural-search profile check researcher
```

`profile check` exits non-zero if required dependencies or artifacts are missing. Missing recommended artifacts are reported without making the profile unusable. This is deliberate. For example, a researcher can still use the normalized corpus before the optional graph and dense cache have been rebuilt, while the readiness report makes that reduced capability visible.

The general environment diagnostic accepts the same profiles:

```bash
neural-search doctor --profile demo
neural-search doctor --profile evaluator
```

## Reproducibility manifests

Any meaningful run can record the environment and artifact state that produced it:

```bash
neural-search profile manifest demo \
  --output reports/reproducibility/demo.json
```

Or with Make:

```bash
make repro-manifest PROFILE=demo
```

The manifest records:

- the selected execution profile;
- the Git commit when available;
- Python and platform information;
- required dependency availability;
- registered artifact paths, kinds, sizes, and timestamps;
- SHA-256 checksums for reasonably sized files;
- declared setup/run commands and resource notes.

It does not record secret values. Very large generated assets are not automatically hashed on every check because doing so would make a simple readiness command unexpectedly expensive.

## Profile journeys

### 1. Evaluate the idea

```bash
make setup
neural-search profile check demo
make demo
```

This path is deterministic and uses committed fixtures. It exercises ontology loading, dataset cards, database seeding, retrieval evaluation, report generation, notebook generation, and search.

### 2. Use Neural Search for research discovery

```bash
make install-researcher
neural-search profile check researcher
NEURAL_SEARCH_PROFILE=researcher make api
make web
```

If the profile check reports a missing full corpus, use the registered repair guidance or obtain the lab's versioned research-asset snapshot. Do not copy an arbitrary graph or embedding cache from another corpus revision without recording that mismatch.

### 3. Build or refresh the corpus

```bash
make install-corpus-builder
neural-search profile check corpus-builder
python scripts/corpus/build_full_corpus.py
python scripts/rebuild_full_corpus_graph.py
python scripts/recompute_embeddings.py --provider dense
```

The corpus builder consumes local raw source payloads. Refreshing those payloads can require network access and source-specific acquisition scripts under `scripts/corpus/`.

### 4. Evaluate retrieval changes

```bash
make install-evaluator
neural-search profile check evaluator
make benchmark
python scripts/build_artifact_manifest.py
```

The canonical query set and qrels are the portable baseline. Full dense and graph ablations additionally require generated evaluation/corpus assets, which are reported as recommended rather than silently assumed.

### 5. Operate the maintainer stack

```bash
make install-full-stack
neural-search profile check full-stack
NEURAL_SEARCH_PROFILE=full-stack make api
make web
```

Use this profile when you actually need the databases, queues, dense retrieval stack, real corpus, graph, and larger analysis surface. It is intentionally not the onboarding default.

## API visibility

A running API exposes read-only runtime information:

- `GET /api/runtime/profiles`
- `GET /api/runtime/status?profile=researcher`
- `GET /api/runtime/artifacts`

These endpoints are designed so the frontend or deployment tooling can say *why* a capability is unavailable instead of failing later on a missing file.

## CI promises

CI checks the portable `demo` and `evaluator` contracts on a clean checkout. An additional adoption smoke job follows the documented `make setup` path and runs the complete demo. If those jobs fail, the repository is not considered portable even if lower-level tests still pass.
