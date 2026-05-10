"""Validate a queue YAML file without starting training.

Checks performed for each run in the queue:
  1. YAML parses correctly and queue structure is valid.
  2. env_name is registered in mujoco_playground.registry.
  3. Every flag name is a recognised CLI flag of train_jax_ppo.py.
  4. Every env_override dotted-key exists in the environment's default_config.

No JAX compilation or GPU initialisation takes place — only the config
dict is loaded, not the environment itself.

Usage:
    python3 learning/validate_queue.py learning/queues/overnight.yaml
    python3 learning/validate_queue.py learning/queues/*.yaml
"""

import pathlib
import sys

import yaml

# ---------------------------------------------------------------------------
# Make project root importable regardless of cwd
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learning.run_queue import parse_queue  # noqa: E402
from mujoco_playground import registry      # noqa: E402

# ---------------------------------------------------------------------------
# Complete set of CLI flags accepted by train_jax_ppo.py
# ---------------------------------------------------------------------------
_KNOWN_FLAGS: frozenset[str] = frozenset({
    # environment / setup
    "env_name", "impl", "vision", "load_checkpoint_path", "suffix",
    "play_only", "use_wandb", "use_tb", "domain_randomization", "seed",
    "env_overrides_file",
    # training budget
    "num_timesteps", "num_videos", "num_evals", "episode_length",
    # PPO hyperparameters
    "reward_scaling", "normalize_observations", "action_repeat",
    "unroll_length", "num_minibatches", "num_updates_per_batch",
    "discounting", "learning_rate", "entropy_cost",
    "num_envs", "num_eval_envs", "batch_size", "max_grad_norm",
    "clipping_epsilon",
    # network architecture
    "policy_hidden_layer_sizes", "value_hidden_layer_sizes",
    "policy_obs_key", "value_obs_key",
    # diagnostics / misc
    "rscope_envs", "deterministic_rscope", "run_evals",
    "log_training_metrics", "training_metrics_steps",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_has_key(cfg, dotted_key: str) -> bool:
    """Return True if every segment of a dotted path exists in a ConfigDict."""
    node = cfg
    for part in dotted_key.split("."):
        try:
            node = node[part]
        except (KeyError, TypeError):
            return False
    return True


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_queue(path: pathlib.Path) -> list[str]:
    """Return a list of human-readable error strings; empty means valid."""
    errors: list[str] = []

    # Step 1: parse queue structure
    try:
        runs = parse_queue(path)
    except Exception as exc:
        return [f"Queue parse error: {exc}"]

    # Step 2: per-run checks
    for run in runs:
        idx = run["idx"]
        env_name = run["flags"].get("env_name", "")
        tag = f"Run {idx:02d} ({env_name!r})"

        # 2a. env_name registered
        if env_name not in registry.ALL_ENVS:
            errors.append(
                f"{tag}: unknown env_name {env_name!r}. "
                f"Known envs: {sorted(registry.ALL_ENVS)}"
            )
            # Can't validate overrides without a valid env — skip rest for this run
            continue

        # 2b. flag names
        for flag in run["flags"]:
            if flag not in _KNOWN_FLAGS:
                errors.append(f"{tag}: unrecognised flag {flag!r}")

        # 2c. env_override keys exist in default config
        if run["env_overrides"]:
            try:
                cfg = registry.get_default_config(env_name)
            except Exception as exc:
                errors.append(f"{tag}: failed to load default config: {exc}")
                continue

            for key in run["env_overrides"]:
                if not _config_has_key(cfg, key):
                    errors.append(
                        f"{tag}: env_override key {key!r} not found in "
                        f"default_config for {env_name!r}"
                    )

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 learning/validate_queue.py <queue.yaml> [...]")
        sys.exit(1)

    all_ok = True
    for path_str in sys.argv[1:]:
        path = pathlib.Path(path_str)
        if not path.exists():
            print(f"ERROR  {path}: file not found")
            all_ok = False
            continue

        errors = validate_queue(path)
        if errors:
            print(f"FAIL   {path}")
            for err in errors:
                print(f"  x  {err}")
            all_ok = False
        else:
            n = len(parse_queue(path))
            print(f"OK     {path}  ({n} run{'s' if n != 1 else ''})")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
