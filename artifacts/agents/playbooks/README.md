# Agent playbooks

Each file here is a step-by-step procedure for one entry in
`artifacts/agents/registry.yaml`. A playbook is written so a Claude Code
subagent (or a human) can follow it directly: concrete commands, concrete
file paths, concrete pass/fail criteria. No playbook here invents new
capability -- each one formalizes a procedure that has already been run
manually at least once, with a real track record (see the cited project
memory in each playbook).

## Running a playbook

1. Read the playbook top to bottom before running anything.
2. Follow its steps in order. Do not skip the gate step.
3. Append exactly one row to `artifacts/agents/ledger.jsonl` via
   `neural_search.agents.ledger.append_ledger_entry()` when done, whether
   the outcome was `ok`, `regression`, `error`, or `no_op`.
4. Write one linked note under `obsidian_vault/11_Agent_Runs/` summarizing
   the run in human-readable form (see existing notes there for the
   frontmatter shape).

## Picking what to run next (the prioritization scoring function)

When more than one agent class is due, score each by:

```text
score = expected_value / estimated_cost.wall_clock_s
```

Where `expected_value` is a rough, explicit judgment call (not a precise
number) based on:

- How long since this agent class last ran (`neural_search.agents.ledger.last_run_for_agent`)
- Whether anything changed that it would care about (a new graph build for
  benchmark-gatekeeper; new file-validation artifacts for
  kg-connectivity-auditor's coverage claims)
- Its `gate` field in the registry: `hard_block` agents that are currently
  overdue should generally win over `soft_review`/`none` ones, since a
  hard-blocked change sitting unverified is the riskiest state this system
  can be in.

In practice this means: **benchmark-gatekeeper almost always wins** when
due -- it is cheap (a few minutes), fast to interpret (pass/fail against a
known number), and guards against the single most expensive mistake this
project can make (a silent ranking regression, already caught 3 times).
Prefer it over starting an expensive, uncertain-yield task
(file-validation-runner, literature-linker) whenever both are technically
due.
