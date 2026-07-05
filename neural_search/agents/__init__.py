"""Lightweight agent-orchestration primitives: a task registry reader and an
append-only run ledger. See `artifacts/agents/registry.yaml` and
`artifacts/agents/playbooks/` for the actual task definitions this supports.

This package deliberately does not implement a scheduler or task queue --
Claude Code subagents already do the "read the registry, decide what's due,
run a playbook" work; this module only gives that work a durable, structured
place to read task metadata from and write its findings to, so a run's cost
and outcome survive past the session that produced it.
"""
