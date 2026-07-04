# Playbook: kg-connectivity-auditor

**Registry entry:** `kg-connectivity-auditor` in `artifacts/agents/registry.yaml`
**Gate type:** `none` (read-only audit; any fix it proposes needs
`benchmark-gatekeeper` before being considered committed).

## Why this exists

Every KG-producing module in this codebase should be in exactly one of
four states: merged into production (`scripts/build_real_corpus_graph.py`'s
`orphaned_layers` list or an earlier direct step), a confirmed side-channel
(read directly by `neural_search/graph/search_features.py` some other way),
genuinely dead code, or an explicitly-deferred orphan with a stated
blocking reason. This procedure already found and resolved all 8
originally-orphaned builders plus one fully dead module
(`neural_search/graph/similarity.py`) across two prior sessions -- see the
`kg_connectivity_audit` project memory for the full history, including the
exact "orphaned" pattern to look for.

## When to run

- Weekly, or immediately after adding any new module under
  `neural_search/graph/` or `neural_search/ingestion/` that builds nodes or
  edges.

## Steps

1. List every module matching `def build_*_kg(` or `def build_*_edges(` or
   `def build_*_nodes(` under `neural_search/graph/` and
   `neural_search/ingestion/` (grep is sufficient; this is a small, stable
   set of modules).
2. For each one, check reachability:
   - **Merged**: is it imported and called (directly, or via
     `orphaned_layers`) in `scripts/build_real_corpus_graph.py`? Grep for
     the function name there.
   - **Side-channel**: is its output artifact (e.g. a `.jsonl` file it
     writes) read directly by
     `neural_search/graph/search_features.py` (e.g. via an
     `_load_*_index()` helper) even though the module itself isn't
     imported into the graph builder? This still counts as reachable --
     confirm by grepping the artifact filename in `search_features.py`.
   - **Dead**: zero callers anywhere outside its own test file and its own
     `__init__.py` re-export. Confirm with a repo-wide grep for the
     function/class name before concluding this -- a name used only in a
     docstring or comment doesn't count as a caller, but check the actual
     call, not just a text match.
   - **Orphan**: none of the above -- built, tested, but genuinely
     unreachable. For each orphan found, check whether reconnecting it
     needs a data prerequisite that doesn't exist yet (e.g. paper nodes
     were the blocker for `paper_node_builder.py`/`citation_builder.py`
     until 2026-07-02/04) -- state the blocker explicitly rather than just
     flagging "not connected".
3. For any side-channel artifact that has **no automated regeneration
   pipeline** (gitignored, hand-run script only) -- flag this as a
   fragility risk even if currently working, per the
   `neurosynth_builder.py`/`composed_kg.jsonl` incident in
   `kg_connectivity_audit` memory (silently went to zero with no error when
   the artifact went missing, until a fallback was added).
4. Also re-check any module previously marked "dead" or "orphan" in an
   earlier audit -- confirm the disposition still holds (new callers can
   appear) rather than assuming a stale finding is still true.
5. Compile findings as a list of
   `{module, status, evidence, proposed_action}`. Do not propose a
   reconnection fix without a specific plan (which builder pattern to
   follow, e.g. `build_paper_nodes_and_links`'s
   `if X not in graph.nodes: continue` scoping pattern) -- a vague
   "should be reconnected" is not actionable.
6. If proposing a code change (reconnecting an orphan), that change must go
   through `benchmark-gatekeeper` before being considered done -- this
   playbook's own job stops at the finding + proposal.
7. Append a ledger entry with `outcome: "ok"` (audit completed, regardless
   of what it found) and `findings` listing each module's status.
8. Write a note under `obsidian_vault/11_Agent_Runs/` with the full table
   of module statuses -- this is the artifact a human or future agent reads
   instead of re-deriving the whole picture from scratch.

## What "done" looks like

Every KG-producing module has a stated, evidenced status. A ledger row and
Obsidian note exist. Nothing is left as "unclear" -- if a status genuinely
can't be determined, that itself is a finding ("evidence inconclusive,
needs X to resolve"), not a silent gap.
