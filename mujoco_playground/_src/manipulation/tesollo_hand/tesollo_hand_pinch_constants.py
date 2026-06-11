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
"""Constants for the static-grasp pinch environment (thumb + index only)."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "tesollo_hand"
SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube_pinch.xml"

# 8 controlled DOFs: thumb (dg_1, joints 1-4) + index (dg_2, joints 1-4).
# Wrist, middle, ring, and pinky are frozen by the scene builder.
N_ACTIVE = 8

JOINT_NAMES = [f"rj_dg_{f}_{i}" for f in (1, 2) for i in range(1, 5)]

FINGERTIP_NAMES = ["rl_dg_1_tip_c", "rl_dg_2_tip_c"]

ACTUATOR_NAMES = [
    "dg_1_j1_rx", "dg_1_j2_rz", "dg_1_j3_rx", "dg_1_j4_rx",
    "dg_2_j1_rx", "dg_2_j2_ry", "dg_2_j3_ry", "dg_2_j4_ry",
]

