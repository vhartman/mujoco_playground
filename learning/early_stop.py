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
triggers a stop.

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

# Fraction of the learned reward gain (best - initial) that must be given back
# for episode/sum_reward to count as "collapsed". 0.5 == lost half its gains.
COLLAPSE_FRAC = 0.5

# Number of consecutive training-metric logs that must show a post-divergence
# collapse before aborting. Any recovery resets this streak.
PATIENCE = 3

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
  ):
    self._warmup_steps = warmup_frac * total_timesteps
    self._kl_ceiling = kl_ceiling
    self._vloss_ratio = vloss_ratio
    self._collapse_frac = collapse_frac
    self._patience = patience

    self._init_reward = None
    self._best_reward = None
    self._vloss_baseline = None  # EMA of v_loss while training is healthy
    self._diverged = False  # latched once an explosion is seen
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
    elif vloss is not None:
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

    if self._collapse_streak >= self._patience:
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
