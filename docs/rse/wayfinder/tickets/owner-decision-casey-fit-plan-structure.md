# Approve the Casey fit-acceptance plan structure (incl. model-spec phase)

- Type: `wayfinder:task` (HITL)
- Status: open
- Assignee: Owner
- Blocked by: none
- Map: [ApJ submission](../map-apj-submission.md)

## Owner decision card

The 2026-08-04 windowed diagnostic run confirmed the fit's blockers are
scientific (railed, unjustified priors; single-component CHIME inadequacy;
window-dependent morphology verdict), not computational. Proposed plan
structure, pending owner approval before the full spec is written to
`docs/rse/specs/plan-casey-joint-fit-acceptance.md` (in the
casey-reviewed-joint-fit worktree):

- Phase 0 — diagnostics of the windowed run (fit numbers done; residual
  notebook panels remaining).
- **Phase 0.5 — model-specification document (new, owner-suggested):**
  generative model derived; prior table with physical justification per
  bound; DM/ToA/tau identifiability analysis; what question model comparison
  answers; injection-recovery validation gate before any real-data quote.
  Supersedes the earlier piecemeal "fix the rails" Phase 3 framing.
- Phase 1 — exact compiled EMG kernel (14.6x prototype; machine-precision
  erfcx port + 1e-8 log-likelihood equivalence gate). Science-neutral;
  proceeds in parallel.
- Phase 2 — seconds-scale exploration tier (binned data, optimizer +
  curvature) for rail/adequacy screening. Science-neutral; parallel.
- Phase 3 — scientific fixes per the model spec (owner-gated decision card).
- Phase 4 — production rerun (both arms, <=15 min target).
- Phase 5 — acceptance, visual vetting, receipts, manuscript queue.

Decision: approve this structure (with Phase 0.5) so the full plan spec and
the model-specification draft are written; or amend phases.

Default if unanswered: none — plan writing waits for this approval (the
owner asked for plan-first working mode 2026-08-04).
