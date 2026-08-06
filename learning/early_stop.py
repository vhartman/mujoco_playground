# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Early stopping for PPO runs that diverge unrecoverably.

A diverged run is wasted compute: starting a fresh seed is more informative
than letting it churn. This module detects the characteristic failure
signature -- a KL explosion -> value-loss explosion -> reward collapse that
never recovers -- and lets the training loop abort early.

Detection is deliberately conservative so that *transient* spikes (which
recover) are not caught; only a sustained, post-divergence reward collapse
triggers a stop. Crucially, the divergence latch is *cleared* once KL/v_loss are
healthy again for a few logs, so a reward dip while KL is healthy -- a curriculum
advancing to a harder level, or ordinary exploration -- never trips a stop. We
stop when things are currently unstable, not when the policy is exploring.

All signals are read from the ``episode/*`` metric namespace, which brax
populates with both episode metrics (``episode/sum_reward``) and PPO-loss
metrics (``episode/kl_mean``, ``episode/v_loss``) when ``log_training_metrics``
is enabled. They therefore all arrive together in the same ``progress_fn``
callback at ``training_metrics_steps`` cadence.

Thresholds live here as module constants on purpose: they are not meant to be
tuned per-run from the command line.
"""

import math

# --- Configuration ---------------------------------------------------------
# A KL above this counts as a KL explosion. Healthy PPO steps are typically
# ~0.01 (brax's adaptive-KL schedule targets exactly that), so 0.1 is ~10x a
# normal step -- clearly pathological, not noise.
KL_CEILING = 0.1

# A value loss above this multiple of its rolling (healthy) baseline counts as
# a value-loss explosion. Relative rather than absolute, so the same rule ports
# across environments with different reward scales.
VLOSS_RATIO = 5.0

# Any finite metric whose magnitude exceeds this is a numeric blow-up, aborted
# immediately like NaN/Inf -- no warmup, no KL/reward gating. This is a pure
# sanity backstop for the "frozen actor (tiny KL) + diverged critic" failure
# where v_loss overflows *toward* infinity without reaching it, so the finite
# guard passes and the KL-health master veto (which only tolerates modest 5x
# curriculum blips) wrongly spares it. Sized from observed data: healthy
# episode/v_loss stays <~1e2 while genuine blow-ups land at ~1e18-1e20, so 1e8
# sits ~6 orders above anything healthy and ~10 below any blow-up -- it cannot
# false-positive yet always catches the pathology.
CATASTROPHIC_CEILING = 1e8

# Fraction of the learned reward gain (best - initial) that must be given back
# for episode/sum_reward to count as "collapsed". 0.5 == lost half its gains.
COLLAPSE_FRAC = 0.5

# Number of consecutive training-metric logs that must show a post-divergence
# collapse before aborting. Any recovery resets this streak.
PATIENCE = 3

# Number of consecutive HEALTHY logs (no KL/v_loss explosion) that clear a
# latched divergence. Without this, a single early transient spike would latch
# `diverged` forever, so any *later* reward dip -- e.g. a curriculum advancing to
# a harder level, or ordinary exploration -- would trip the collapse check even
# though KL is completely healthy. We only stop when things are *currently*
# unstable, not when the policy is merely exploring.
RECOVERY_PATIENCE = 3

# Fraction of num_timesteps to skip before the KL / v_loss / reward heuristics
# may fire. The NaN/Inf guard is always active, regardless of warmup.
WARMUP_FRAC = 0.1

# EMA smoothing factor for the value-loss baseline.
_EMA_ALPHA = 0.1


class EarlyStopException(Exception):
  """Raised to abort a diverged run early (caught by the training script)."""


class EarlyStopper:
  """Stateful divergence detector, updated once per training-metrics log.

  Usage::

      stopper = EarlyStopper(total_timesteps)
      # inside progress_fn(num_steps, metrics):
      reason = stopper.update(num_steps, metrics)
      if reason is not None:
        raise EarlyStopException(reason)
  """

  def __init__(
      self,
      total_timesteps,
      kl_ceiling=KL_CEILING,
      vloss_ratio=VLOSS_RATIO,
      collapse_frac=COLLAPSE_FRAC,
      patience=PATIENCE,
      warmup_frac=WARMUP_FRAC,
      recovery_patience=RECOVERY_PATIENCE,
  ):
    self._warmup_steps = warmup_frac * total_timesteps
    self._kl_ceiling = kl_ceiling
    self._vloss_ratio = vloss_ratio
    self._collapse_frac = collapse_frac
    self._patience = patience
    self._recovery_patience = recovery_patience

    self._init_reward = None
    self._best_reward = None
    self._vloss_baseline = None  # EMA of v_loss while training is healthy
    self._diverged = False  # set on explosion, cleared after sustained health
    self._healthy_streak = 0  # consecutive logs with no explosion
    self._collapse_streak = 0

  def update(self, num_steps, metrics):
    """Return a reason string if training should stop, else None."""
    # 1) NaN / Inf guard -- always active, immediate abort.
    for key, value in metrics.items():
      try:
        fv = float(value)
      except (TypeError, ValueError):
        continue
      if not math.isfinite(fv):
        return f"non-finite metric {key}={value} at step {num_steps}"
      if abs(fv) > CATASTROPHIC_CEILING:
        return f"catastrophic metric {key}={value} at step {num_steps}"

    kl = metrics.get("episode/kl_mean")
    vloss = metrics.get("episode/v_loss")
    reward = metrics.get("episode/sum_reward")

    if reward is not None:
      reward = float(reward)
      if self._init_reward is None:
        self._init_reward = reward
      if self._best_reward is None or reward > self._best_reward:
        self._best_reward = reward

    # During warmup only learn the v_loss baseline; never fire heuristics.
    if num_steps < self._warmup_steps:
      if vloss is not None:
        self._vloss_baseline = self._ema(self._vloss_baseline, float(vloss))
      return None

    # 2) Detect an explosion (leading indicator), then latch it.
    exploded = False
    if kl is not None and float(kl) > self._kl_ceiling:
      exploded = True
    if (
        vloss is not None
        and self._vloss_baseline is not None
        and float(vloss) > self._vloss_ratio * max(self._vloss_baseline, 1e-8)
    ):
      exploded = True

    if exploded:
      self._diverged = True
      self._healthy_streak = 0
    else:
      # Sustained health clears a latched divergence: a long-ago transient spike
      # must not keep `diverged` armed while KL/v_loss are now fine. This is what
      # stops curriculum/exploration reward dips from tripping the collapse check.
      self._healthy_streak += 1
      if self._healthy_streak >= self._recovery_patience:
        self._diverged = False
        self._collapse_streak = 0
      if vloss is not None:
        # Only track the baseline while healthy, so it doesn't chase a spike.
        self._vloss_baseline = self._ema(self._vloss_baseline, float(vloss))

    # 3) Confirm a sustained reward collapse, gated on prior divergence.
    collapsed = False
    if (
        self._diverged
        and reward is not None
        and self._best_reward is not None
        and self._init_reward is not None
    ):
      improvement = self._best_reward - self._init_reward
      if improvement > 0:
        threshold = self._best_reward - self._collapse_frac * improvement
        collapsed = reward < threshold

    self._collapse_streak = self._collapse_streak + 1 if collapsed else 0

    # Final gate, per the directive "stop when unstable, not when exploring":
    # NEVER fire while KL is currently healthy. A latched divergence + reward dip
    # with healthy current KL is a curriculum advancing to a harder band (reward
    # legitimately drops) or ordinary exploration -- not instability. Only stop a
    # policy that is *currently* unstable (KL still above the ceiling). This is
    # robust to the failure mode where intermittent v_loss blips (the critic
    # re-adapting each time the curriculum hardens) keep `diverged` armed while
    # the reward sits lower at the harder level with KL perfectly healthy.
    kl_currently_unstable = kl is not None and float(kl) > self._kl_ceiling
    if self._collapse_streak >= self._patience and kl_currently_unstable:
      return (
          "divergence + sustained reward collapse: episode/sum_reward="
          f"{reward:.3f} vs best={self._best_reward:.3f} for"
          f" {self._collapse_streak} consecutive logs (kl_mean={kl},"
          f" v_loss={vloss}) at step {num_steps}"
      )
    return None

  @staticmethod
  def _ema(prev, value, alpha=_EMA_ALPHA):
    return value if prev is None else (1 - alpha) * prev + alpha * value
