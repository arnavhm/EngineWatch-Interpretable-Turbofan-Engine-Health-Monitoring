# EngineWatch Stability Gate

Referenced from the AeroGraph vision page as its start condition. This is
the actual checklist — not a feeling, not a date. AeroGraph doesn't get
real attention until every box below is checked, verified the same way
everything else this session was: real output, not a claim.

## The checklist

- [ ] **Frontend Level 2 redesign shipped.** Fleet Command + Engine
      Drill-Down both live at enginewatch.tech, not just committed as
      untracked files. Browser fresh-viewer check done by Arnav personally.
- [ ] **`ci-live-gate.yml` green for 5 consecutive scheduled runs**, not
      just the one manually-triggered run that proved it works. Five
      real runs, unattended, actually catching nothing wrong.
- [ ] **Zero open Risk Register items with Status = Active and Impact above
      Low.** 'Monitoring' status doesn't count against this — an ongoing
      watched pattern with a real mitigation already in place is a healthy
      steady-state, not a blocker.
- [ ] **`.agents/test-prompts.md` all 5 tests run at least once** since
      the last `AGENTS.md` change, with real pass results shown.
- [ ] **The Caddy `try_files` fix for `/engine/:id` hard-refresh** is
      shipped (noted as a small pending ops item from the paused frontend
      session).
- [ ] **The two minor frontend findings are resolved**: the duplicate
      `/predict` fetch, and the missing form label (a11y).

## What this checklist is *not*

Not a gate on code quality in the abstract — EngineWatch's backend/ML core
is already in good shape. This is specifically about whether the *current
iteration* is closed out cleanly enough that starting something new
doesn't mean context-switching away from loose ends. A half-finished
redesign plus a brand-new knowledge-graph project running in parallel is
exactly the kind of two-things-at-once situation that already caused
friction once this session (frontend chat vs. this coordination chat).

## When this is fully checked

Update this file, then update the AeroGraph Notion page's gating section
to point at the real date instead of "not yet drafted as a formal
checklist." That's the actual green light — not August 1st on a calendar.
