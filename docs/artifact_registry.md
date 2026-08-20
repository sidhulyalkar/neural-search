# Artifact Registry

Neural Search depends on scientific assets with very different portability and lifecycle expectations. Treating all of them as ordinary repository files caused recurring fresh-clone failures and made it difficult to tell whether a missing file was a bug, an optional capability, or an expected local research asset.

The runtime artifact registry in `neural_search/runtime/catalog.py` is the first-class contract for that boundary.

## Artifact kinds

| Kind | Meaning | Fresh-clone expectation |
| --- | --- | --- |
| `committed_fixture` | Small deterministic inputs used by demos/tests | Must exist |
| `frozen_evaluation` | Versioned scientific evaluation inputs intended to travel with the code | Must exist |
| `generated_local` | Large/downloaded/derived research assets | May be absent; capability checks must report this explicitly |
| `derived_report` | Recomputable summaries of current local state | May be absent or regenerated |

A generated asset becoming scientifically important does not automatically mean it should be committed to Git. Large artifacts should instead be versioned and distributed through an artifact-release mechanism once one is established.

## What each registry entry should answer

Every registered artifact has:

- a stable artifact ID;
- a repository-relative expected path;
- a lifecycle kind;
- a human-readable purpose;
- a producer description when the asset is generated;
- a repair/rebuild command when one canonical command exists.

Do not add an unverified repair command simply to make the registry look complete. If several acquisition steps or external credentials are required, leave the command unset and document the real pipeline.

## Inspecting artifacts

```bash
neural-search artifacts list
neural-search artifacts status
neural-search artifacts status full_corpus_v09 production_graph
```

The output distinguishes `missing_portable_asset` from `missing_generated_asset`. The first is generally a repository/release problem. The second can be a normal state for a smaller execution profile.

The API exposes the same information through `GET /api/runtime/artifacts` so user interfaces and deployments can use one source of truth.

## Artifact dependency rules

1. **Transport code should not guess paths independently.** If a new first-class runtime asset becomes required, register it and make the consuming capability declare its profile relationship.
2. **Tests must not require generated-local assets in a clean checkout** unless the CI job explicitly constructs or downloads them first.
3. **Scientific reports should name their upstream corpus/evaluation state.** A report without a reproducible input identity is weaker evidence than a report with the same metrics plus provenance.
4. **Do not silently substitute an older artifact revision.** A v07 graph is not a harmless fallback for a v09 corpus if the resulting metrics or evidence can change.
5. **Generated artifacts must not upgrade evidence tiers.** A graph edge, LLM judgment, or derived linkage remains the evidence type produced by its pipeline.

## Relationship to the scientific artifact manifest

`scripts/build_artifact_manifest.py` and `reports/eval/current_artifact_manifest.json` summarize the *contents* of important scientific artifacts, such as corpus row counts, graph node/edge counts, qrels tiers, and literature-link coverage.

The runtime registry answers a different question: *what artifact is this, who needs it, should it be present here, and how can the environment tell that it is missing?*

Both are useful:

- runtime registry: capability/dependency contract;
- scientific artifact manifest: measured current-state summary.

The long-term direction should connect them rather than replacing one with the other.

## Adding a new artifact

When a new pipeline introduces a durable asset:

1. decide whether it is a fixture, frozen evaluation input, generated local asset, or derived report;
2. add an `ArtifactSpec` to `neural_search/runtime/catalog.py`;
3. attach it to the appropriate execution profile as required, recommended, or produced;
4. add a repair command only if that command actually reconstructs the expected artifact;
5. add or update tests proving portable profiles do not accidentally depend on it;
6. include the artifact in reproducibility/reporting surfaces when it affects scientific results.

## Next infrastructure step: versioned artifact releases

The current registry makes absence explicit but does not yet download large assets. A future artifact backend should support immutable versioned bundles, checksums, metadata, and optional remote acquisition, for example:

```text
artifact id + version
        ↓
manifest/checksum
        ↓
remote object store or release
        ↓
local cache
        ↓
profile readiness
```

That layer should be content-addressed or otherwise immutable enough that another lab can retrieve the exact corpus/graph/index combination associated with a reported experiment. It should not become an automatic "download whatever is newest" mechanism, because reproducibility requires pinning state rather than chasing it.
