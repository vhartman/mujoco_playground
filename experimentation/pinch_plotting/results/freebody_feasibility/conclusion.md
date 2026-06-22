# Free-body cube — feasibility (NEGATIVE, both variants ruled out)

**Free-body cube is NOT a viable lever for the 2-finger pinch.** Two variants tried:
1. lifting/mass — frozen wrist can't lift the cube off the floor (mass never bites);
2. moving-complexity (free cube on floor, squeeze to force) — **catastrophically
   unstable**: pinching a free object between two opposing fingers ejects it.

**Use cube *position* + *size* DR on the welded cube instead** to get "the cube
varies so open-loop maps fail" without the instability.

## Full de-risk run (weld_cube=false, force.magnitude, 300M steps)
`logs/TesolloCubePinch-*-freecube_derisk`
- Reward *collapses* with training: 40M→2.7, 300M→**0.2** (diverging, not converging).
- Cube z end mean ≈ **88 m** over 10 rollouts — the cube is launched/ejected by the
  pinch (and a vertical launch isn't even caught by the xy `drift` check). Final
  |force_error| ≈ 3.4 N.
- Physically: opposing-finger pinch on an unconstrained body is degenerate; there is
  no palm/wrist to stabilise it.

## Evidence (smoke-train: weld_cube=false, force.magnitude, 40M steps)
`logs/TesolloCubePinch-20260618-203859-freecube_smoke`
- Fingertips grip at z ≈ 0.032–0.053 m, cube at 0.035 m → the frozen wrist **cannot
  lift the cube clear of the floor**; over 6 rollouts the cube was >5 mm above rest
  only 5% of steps. So gravity/mass is borne by the floor → **mass is not a usable
  latent**. The lifting/airborne idea is out (would need an unfrozen wrist).
- A free cube is dynamic under the squeeze (a 0.3 squeeze pops it to z≈0.086 in one
  step) and the static force-target objective degrades (40M-step reward ~2.7 vs
  ~15–22 welded; |force_error| ≈ 3.1 N). This is the intended **moving complexity**:
  the same joint pose no longer maps to a fixed force, so an open-loop policy fails.

## Decision
- Drop everything lifting/mass related (done: removed `cube_start_z`, the `dropped`
  metric/termination). The free cube starts at its home (floor) pose and is free to
  shift/rotate; episodes are fixed-length and reward-driven (no drop termination).
- Keep `weld_cube` as a task lever: **welded (open-loop-prone) vs free (moving)** is
  the dynamic counterpart to the size/position DR geometric latents.
- Force-sensing experiments: E1 cube-size DR, E2 cube-position DR (both welded), and
  E3 free-cube (moving), each x action mode x bundles x 3 seeds.

The free cube force task is hard (smoke), so E3 needs full-length training to judge
whether force-informed bundles handle the dynamics better than blind ones.
