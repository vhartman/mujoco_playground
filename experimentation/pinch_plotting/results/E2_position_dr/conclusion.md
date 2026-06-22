# E2 — cube initial-position DR

**Verdict: NULL.** Randomizing the cube's (unobserved) initial position does **not**
create a force-sensing testbed for this frozen-wrist pinch — it's a *reach* problem,
not a *force* problem. Force-informed bundles do not beat blind ones, and randomized-
position training confers no robustness.

## Setup
48 runs (all passed): `{none, baseline, proprio.target, force.magnitude}` ×
{fixed, randomized cube_pos ±1.5 cm} × {delta, absolute} × 3 seeds. Welded cube.
Eval: deterministic cube x-offset sweep (−0.015…0.015 m), DV = hold-window
|force_error|. Figure: `rollout_data/force_error_vs_pos_e2.png`.

## Evidence
- **No fixed-vs-randomized separation** — randomized-position training did not make
  policies position-robust. E.g. delta `baseline`: fixed `[2.13, 1.78, 0.52, 1.95,
  2.13]` ≈ rand `[1.90, 1.85, 0.50, 1.83, 2.47]` N across offsets.
- **No blind-vs-informed separation** — all bundles are U-shaped (min ~0.5 N at
  offset 0, ~2–3 N at ±0.015), in both action modes; `force.magnitude` is among the
  worst, not best.
- **Mechanism (decisive):** at +0.015 m a trained policy reads
  `f_thumb = f_index = 0 → effective_force = 0` — **both fingers miss the cube**. The
  frozen-wrist fingertips are at fixed xy, so an offset cube leaves their grasp. This
  is a binary contact/reach failure; no amount of force sensing or training fixes it.

## Implication
Cube **position** moves *where contact must happen* (reach), which this hand can't
follow — not *how much force a given pose yields*. Use cube **size** (E1): it changes
the force-from-pose map with the cube staying centred (no reach failure), and earlier
showed real separation. Position DR is dropped as a force-sensing lever.
