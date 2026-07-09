---
name: execute-claude-brief
description: Use this skill whenever Arnav pastes a structured, phase-gated implementation brief authored by Claude for the EngineWatch project's backend, ML pipeline, API, or deploy work — briefs typically contain numbered phases, explicit stop-and-report gates, or file-level instructions written for Antigravity to execute. Also trigger on any multi-step backend/ML/deploy change, even if it isn't explicitly labeled a "brief." Do NOT use this for frontend/React/UI work at /var/www/enginewatch — that's the enginewatch-frontend-builder skill's job; defer to it instead.
---

# Execute Claude Brief

This skill governs *how* to work through a brief Claude wrote for this project —
pacing, gating, and what counts as a completed step. It does not restate the
underlying environment/verification rules; those live in `AGENTS.md` at the
project root and apply regardless of whether this skill is active. Read
`AGENTS.md` first if you haven't already this session.

## Scope

This skill covers backend, ML pipeline, API, and deploy briefs. Frontend
briefs (React/Vite/Tailwind at `/var/www/enginewatch`) are handled by the
`enginewatch-frontend-builder` skill instead — it has its own domain rules
(component reuse, Tailwind-only, bundle discipline) layered on top of the
same `AGENTS.md` invariants this skill also defers to.

## Recognizing a brief

Look for: numbered phases, a section that says something like "stop and
report" or "wait for confirmation," explicit file paths with line-level
instructions, or a stated acceptance/gate condition per step. Claude's briefs
for this project are deliberately literal and no-guesswork — if a step seems
to require inventing an approach Claude didn't specify, that's a signal to
stop and ask rather than improvise.

## Execution pacing

1. Work exactly one phase (or one independently-testable unit of work) at a
   time. Never batch multiple unverified changes into a single report — if
   you touch three functions across two phases before stopping, Claude can't
   tell which change caused which result.
2. At the end of each phase, stop. Do not start the next phase until you've
   produced a gate report (format below) and received explicit go-ahead.
3. If the brief doesn't define explicit phase boundaries, create implicit
   ones: one endpoint change, one function change, one migration step — each
   gets its own report, not a combined summary at the end.

## Gate report format

Reuse the format from `AGENTS.md` Section 10a exactly, at the tier
Section 10 calls for:

```
## Task: <name>

### Changes
<real diff, or exact file list with real content>

### Verification run
<exact commands executed>
<exact raw output>

### Environment check
<output of `which python` + `pip show` per AGENTS.md Section 2>

### Status
PASS / FAIL against each relevant gate
```

A phase is not complete without this. "This phase is done" with no attached
report is not a valid gate report — say so explicitly if you skipped a step
in it rather than omitting the section.

## Conflict handling

If anything in the brief conflicts with an `AGENTS.md` invariant — e.g. it
implies a silent fallback, skips venv verification, lets `dataset_id`
default (including via an implicit "active config" lookup, not just a
function default), or asks you to skip the deploy verification gate — stop
and surface the conflict as its own line in your response. Do not silently
follow the brief over the invariant, and do not silently follow the
invariant while ignoring what the brief asked for. Name the conflict and let
Arnav or Claude resolve it.

This applies even when a request is explicit and technically permitted. If a
prompt comes directly from Arnav rather than through one of Claude's briefs,
and it requires a design or architecture decision Claude hasn't weighed in
on — which statistical method to use, how to combine two signals, whether an
analysis is valid for a given dataset — that's a decision this skill isn't
authorized to make silently just because the ask was clearly worded.
Explicit instructions can still imply an undecided design choice underneath
them; naming that choice out loud is not optional. Flag it and ask, or
suggest routing it through Claude first, rather than picking a reasonable-
looking implementation and reporting it as done.

## Ambiguity handling

If a brief step is genuinely underspecified (not just "harder than expected"
but actually missing information needed to proceed correctly — e.g. it
references a file or config value that doesn't match what's actually in the
repo), stop and ask rather than picking a plausible interpretation and
reporting it as if it were specified. This project has a documented history
of plausible-sounding but wrong assumptions passing as "confirmed" — don't
add to it.

## Scope discipline

Do only what the current phase asks. If a phase's fix reveals an adjacent
issue worth flagging, mention it in the report as an observation — don't fix
it inline unless the brief says to. Scope expansion without instruction is
out of bounds per `AGENTS.md` Section 8.
