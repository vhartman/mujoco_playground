# E1 — cube size DR

**Verdict: POSITIVE (action-mode dependent).** Under unobserved cube-**size**
randomization with **delta** actions, force sensing pays off: a target-only blind
policy (`none`) is stuck at an irreducible error floor while force-/q-observing
policies adapt. Under **absolute** (target-joint) actions the separation collapses.

## Setup
48 runs (all passed): `{none, baseline, proprio.target, force.magnitude}` ×
{fixed, randomized cube_size [0.85,1.15]} × {delta, absolute} × 3 seeds. Welded cube
(stays centred & floor-resting → no reach failure, unlike E2). Eval: cube-size sweep,
DV = hold-window |force_error|. Figure: `rollout_data/force_error_vs_size_e1.png`.

## Evidence (mean |force_error| N, in-distribution 0.85–1.0)
**delta, randomized (the test):**
- `none` (blind): **~1.70–1.82** floor — cannot get below it; it lacks the signal to
  disambiguate size.
- `force.magnitude`: **~0.34–0.43** — best; clearly uses measured force.
- `baseline` (q): **~0.6–1.0** — also adapts (q closes the loop via joint-stall, as
  in the earlier open-loop study).
- gap `none − force.magnitude` ≈ **1.3 N** = value of force feedback.

**delta, fixed:** all bundles degrade off-nominal (min at 1.0, ~4–6 N at 1.15) → they
hard-code the force_target→pose map; randomization is necessary to expose feedback.

**absolute (both fixed and randomized):** bundles cluster ~1.3–1.6 N with little
separation — target-joint actions make the task more open-loop, hiding force sensing
(consistent with the earlier action_mode finding that absolute ≈ open-loop).

## Caveats
- `proprio.target` underperforms in delta+randomized (~1.6 N, worse than `baseline`)
  — higher seed variance; the clean contrast is `none` vs `baseline`/`force.magnitude`.
- The 1.15 edge spikes for all bundles (training-distribution boundary, data-sparse).

## Takeaway for the study
Cube **size** is the working force-sensing lever (vs E2 position = null reach problem,
and free-body = ruled out). A policy is **open-loop iff it observes neither q nor
force** (`none`); q or measured force makes it closed-loop — but only when the action
space is **delta**. The action output mode is a genuine moderator of whether force
sensing is used.
