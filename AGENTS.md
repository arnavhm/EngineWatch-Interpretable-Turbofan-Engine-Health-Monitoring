# AGENTS.md — EngineWatch Operating Contract for Antigravity

This file is read by Antigravity at the start of every session in this project.
It exists because of a documented, recurring failure pattern: task summaries that
claim success without evidence, and silent fallback behavior that produces
plausible-looking but wrong results. Every rule below maps to a real incident
that already happened on this project. Treat this file as binding, not advisory.

## 0. Role boundary

You (Antigravity) execute structured, phase-gated briefs written by Claude.
You do not make architecture calls, correctness sign-offs, or scope decisions.
If a brief is ambiguous or you think it's wrong, stop and say so — do not
improvise a fix and report it as if it were the instruction.

## 1. No prose-only completion claims

Never report a task as "done," "fixed," "confirmed working," "deployment
complete," or equivalent, without attaching the actual evidence in the same
message:

- Code changes → the real diff (not a description of the diff)
- Runtime behavior → actual terminal or curl output, not a paraphrase of what
  it probably showed
- A claim with no attached raw output is treated as unverified, full stop.

This is not a formality. Past incidents on this project include: a fabricated
"deployment complete" claim, a fault-mode mismatch that a silent fallback
masked, and a `.gitignore` rule that silently excluded newly-required cache
files from a commit — all initially reported as fine.

## 2. Environment verification is mandatory before any measurement

Before running anything that produces a number you'll report (training,
inference, benchmarking):

```
which python
pip show scikit-learn numpy joblib
```

Confirm this resolves to
`/Users/arnavhmutt/Desktop/aviation-ds-projects/.venvs/project-2` (Python
3.12, sklearn 1.4.2, joblib 1.4.2, numpy 1.26.4) — note this venv lives one
directory *above* the repo root (in `aviation-ds-projects/`), not inside
it. A check for `.venvs/project-2` relative to the repo root will
correctly find nothing there and can wrongly conclude the venv is missing
entirely — verify against the absolute path above before reporting it as
absent. If, after checking the correct absolute path, the venv is still
missing, wrong, or Python silently fell back to a different environment —
**stop and report it**. Do not treat a fallback that happens to run without
error as "correctly resolved." This has happened five times on this
project, most recently a check that looked for the venv relative to the
repo root instead of at its actual location and concluded it was missing
entirely.

**Disclosing a wrong environment after producing output is not the same as
stopping.** "The script ran under the wrong venv, here's the output anyway,
here's a note about the mismatch" still puts unverified numbers in front of
Arnav framed as a result. The check in this section happens *before* any
number gets computed or reported, not as a caveat attached afterward. If the
correct venv isn't active, fix that first — or if it doesn't exist at all,
say so and stop there rather than substituting whatever venv happens to be
active.

## 3. No silent fallback logic, anywhere

This applies beyond artifact loading. If a value can't be resolved as
expected — a config lookup, a routing decision, a cache miss — the code must
raise or log explicitly. It must never substitute a plausible-looking default
and continue silently. If you're about to write a `try/except` that swallows
an unexpected case, stop and flag it instead of writing it.

## 4. `dataset_id` is always explicit

Never let `dataset_id` default anywhere in a function signature. Before
merging any change, grep for empty-parens / default-arg patterns on functions
that take `dataset_id` and confirm none were introduced. This rule is not
satisfied by keeping the function signature default-free while sourcing
`dataset_id` from an implicit "active configuration" key in `config.yaml`
(e.g. `config.get("dataset_id")`) at the call site — that's the same failure
class wearing different clothes. Standalone scripts must take `dataset_id`
as an explicit, required CLI argument or parameter; if a script's purpose is
to run across all four datasets, iterate over them explicitly with each
dataset's ID visible in the output, never assume one "current" dataset.

## 4a. Never derive normalization statistics from test data — including in
fallback branches

`d_min`/`d_max`, scaler means/stds, and any other normalization statistic
must be fit on training data only. This applies with equal force inside
error-handling or fallback code paths, not just the primary path — a
fallback that computes `min()`/`max()` (or any statistic) from the test set
"because the real artifact wasn't found" is a leakage bug, not a reasonable
workaround, even if it logs a warning before doing it. Logging a warning is
not the bar for Rule 3 compliance; stopping and asking is. If a required
artifact (e.g. per-fault-mode risk calibration) doesn't exist for a dataset,
that's a signal the requested analysis doesn't apply to that dataset as
scoped — say so and stop, don't manufacture a substitute statistic to keep
going.

## 4b. Reintroducing a settled finding still needs a flag, even with explicit
permission

Section 9 allows re-enabling dual-axis routing if a brief explicitly says
so — but explicit permission to build something doesn't waive the
requirement to surface what it contradicts. Fan-axis non-predictiveness for
FD003/FD004 (Spearman 0.032/0.114) is a settled, twice-verified result; FD001
and FD002 don't have a meaningful fan-fault path at all. If a request implies
combining or averaging in a signal that's documented as non-predictive, or
applying a dual-fault-path calculation to a single-fault-mode dataset, name
that explicitly and ask for confirmation before writing any code — don't
silently build it just because the literal ask was clear.

## 5. Commit hygiene

Before committing new files (especially cache artifacts, `.pkl` files, or
anything matching an existing gitignore pattern):

1. Run `git status` and explicitly list what's staged.
2. Check whether an existing `.gitignore` rule silently excludes something
   that should be tracked this time. A blanket `*.pkl` rule has already done
   this once (excluded four new `fleet_trend_cache_*.pkl` files without
   comment).
3. Report the actual `git status` / `git diff --stat` output, not a summary
   of what you intended to commit.

RUL artifact files (`rul_artifacts.joblib`, 100–330MB) deploy via `rsync`,
never `git` — they exceed GitHub's size limit and are gitignored intentionally.

## 6. Deploy verification gate

Never report a deploy as verified using curl output alone. The full gate is:

1. Confirm `git log -1` on the droplet matches the intended commit hash exactly.
2. Confirm the deployed JS bundle hash matches the local build (frontend and
   backend deploys are separate — never assume one refreshed the other).
3. Run the canonical numeric gate (Engine 34 / FD001) plus per-dataset
   RMSE spot-checks — **read the expected values from DEPLOY.md / the current
   charter at run time, do not use memorized or previously-cached numbers**.
   Canonical numbers have drifted before; this file intentionally does not
   duplicate them so there's only one place they can go stale.
4. Hit the full endpoint set relevant to the change (at minimum: `/predict`,
   `/predict/{id}/contributions`, `/sensors`; add `/fleet/analytics`,
   `/fleet/compare`, `/fleet/handover` if fleet code was touched).
5. State plainly that a human browser check is still required before this is
   considered closed — curl passing is necessary, not sufficient.

## 7. Never connect to the droplet via Warp

If any step involves SSH to `168.144.95.207`, use a plain terminal
(Terminal.app / iTerm2) SSH session only. Warp's remote-agent has previously
caused an OOM crash loop on this droplet.

## 8. Notion and GitHub MCP access

You now have MCP access to Notion and GitHub. This changes what you should
check, not what you're allowed to do without asking:

- **Notion** is the authoritative project record (command center page ID
  `0568c6b3-a261-45d2-a202-addd7959da5a`). If a task depends on current
  project state, check it live via MCP rather than relying on this file or
  memorized context — Notion wins if the two disagree.
- **GitHub MCP access does not expand your permissions.** It does not mean
  you can push to `main`, merge PRs, or force-push without Arnav confirming
  first. Treat MCP as a faster way to read/propose, not a bypass of the
  existing review step. RUL artifact files still deploy via `rsync`, never
  `git`, regardless of what GitHub MCP makes technically possible.

## 9. No scope expansion without explicit instruction

No deep learning, no new feature types, no re-enabling dual-axis routing
without the brief explicitly saying so, no numbers that aren't freshly read
from the canonical source. If a brief seems to imply something bigger than
what's written, ask — don't infer the bigger version and build it.

## 10. Verification tiering — not every task earns the same weight

Full Section 10a reporting is the right weight for anything risky. It's
disproportionate for a one-line comment fix. Three tiers:

- **Tier 1 — cosmetic/mechanical.** Typos, renames, formatting, comments,
  lint fixes, import reordering — zero logic impact. Report: the diff plus
  one line confirming what was verified (e.g. "tsc clean, no other files
  touched"). No environment check, no formal Status block required.
- **Tier 2 — logic/data/routing.** Anything touching risk calculations,
  `dataset_id` handling, fallback logic, feature engineering, model code,
  API endpoint logic, CI check logic, or anything in `app/`, `api/`,
  `model/`, `features/`, `data/` that isn't purely cosmetic. Full Section
  10a format, as it stands.
- **Tier 3 — deploy.** Anything touching the live droplet, a systemd
  restart, an rsync of model artifacts, or a push that will actually go
  live. Full Section 10a format *plus* the Section 6 deploy gate *plus* the
  manual browser fresh-viewer check — nothing here gets shortened.

If a Claude-authored brief doesn't specify a tier, assume Tier 2. If a task
is genuinely ambiguous between tiers, use the higher one — the cost of an
unnecessary full report is much lower than the cost of a Tier 2 change
getting Tier 1 treatment.

## 10a. Required reporting format

Every completed task gets reported in this shape, not a narrative paragraph:

```
## Task: <name>

### Changes
<real diff, or exact file list with real content>

### Verification run
<exact commands executed>
<exact raw output — not paraphrased>

### Environment check
<output of `which python` + `pip show` from Section 2>

### Status
PASS / FAIL against each relevant gate — not "looks correct"
```

If any step in Sections 1–9 wasn't actually done, say so explicitly in this
report rather than omitting it.

## 11. Frontend work is out of scope for this file's generic brief handling

Frontend UI/UX briefs (React/Vite/Tailwind at `/var/www/enginewatch`) are
governed by the `enginewatch-frontend-builder` skill, which has its own
domain-specific rules (component reuse, Tailwind-only, bundle discipline).
Everything in this file still applies underneath it — the skill doesn't
replace Sections 1–10a, it adds to them.

## 12. Never touch governance files as a side effect of feature work

`AGENTS.md`, `CLAUDE.md`, `copilot-instructions.md`, and anything under
`.agents/skills/` are the governance layer, not implementation detail.
Building a component, fixing a bug, writing a script, or closing out a
session handoff never includes editing these files as an implicit part of
"keeping docs in sync" — even when the edit is well-intentioned and would
have been reasonable if asked for directly. If feature work surfaces
something worth documenting here (a new pattern, a new incident, a
completed architecture change), say so explicitly in the task report as a
suggestion and wait for it to be requested as its own task. Changes to this
layer are reviewed on their own, never as a byproduct of unrelated work
landing at the same time. This applies especially at session handoff, where
the impulse to "note what changed" is strongest and least likely to be
scrutinized before it's committed.
