# VES-Modeling Agent Guide

## Scope

VES-Modeling is a verifier-first computational modeling engine. Keep changes
inside the executable modeling loop: candidate generation, isolated execution,
independent host verification, judgment, search, and verified delivery.
Do not add full competition workflow, document generation, orchestration, or a
universal modeling framework here.

## Trust boundaries

- Candidate claims are untrusted; metrics must come from the host verifier.
- Hidden labels never enter prompts, candidate-visible directories, containers,
  public artifacts, logs, or records.
- Real LLM-generated code runs only through the Docker sandbox. Local execution
  is limited to explicitly trusted fixtures and tests.
- Use the published `verified-executable-search` dependency; do not vendor or
  copy Core search control flow into this repository. Record Core findings in
  `docs/bug-log.md`; submit a tested upstream PR only
  for small, local bugs, and use an upstream Issue for architectural problems.
- Never commit API keys, tokens, private datasets, `team/`, `data/`, or `runs/`.

## Implementation rules

- Read related code and callers before editing.
- Prefer the smallest concrete change and preserve public interfaces.
- Do not introduce speculative factories, registries, managers, or universal
  domain abstractions before repeated real-domain requirements exist.
- Add regression tests for behavior changes and known bugs.

## Definition of done

- Relevant tests pass; the default non-Docker CI command is
  `pytest -m 'not docker'`.
- Docker/security-boundary changes are also verified with Docker tests when the
  daemon is available.
- `ruff check .` passes and no unrelated behavior or dependencies change.
- Public result/status names do not overclaim verification.
- Documentation, examples, and all callers are updated for public API changes.
- The diff contains no secrets, hidden truth, private data, or internal team
  files.

## Team message protocol

When a local team workspace is configured, use its generated role `AGENTS.md`
for identities and transport. Message kinds are `task:`, `result:`, `finding:`,
`review:`, `question:`, `handoff:`, and `notice:`. Only an assigned `task:` or
`handoff:` authorizes implementation; include task id, changed files, exact
test evidence, risks, and commit in completion reports.
