# Backward-compatibility shim — import from pick_and_place directly.
from mujoco_playground._src.manipulation.tesollo_hand.pick_and_place import (  # noqa: F401
    PickAndPlace,
    PickAndPlace as PickAndPlaceBase,
    default_config,
    domain_randomize,
)
