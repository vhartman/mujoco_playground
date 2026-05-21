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
"""Constants for leap hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "tesollo_hand"
SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx_pick_and_place.xml"

NQ = 26
NV = 26
NU = 26

WRIST_JOINT_NAMES = [
    "rj_wrist_0_1",
    "rj_wrist_0_2",
    "rj_wrist_0_3",

    "rj_wrist_1_1",
    "rj_wrist_1_2",
    "rj_wrist_1_3",
]

JOINT_NAMES = [
    "rj_wrist_0_1",
    "rj_wrist_0_2",
    "rj_wrist_0_3",

    "rj_wrist_1_1",
    "rj_wrist_1_2",
    "rj_wrist_1_3",

    # thumb
    "rj_dg_1_1",
    "rj_dg_1_2",
    "rj_dg_1_3",
    "rj_dg_1_4",
    # index
    "rj_dg_2_1",
    "rj_dg_2_2",
    "rj_dg_2_3",
    "rj_dg_2_4",
    # middle
    "rj_dg_3_1",
    "rj_dg_3_2",
    "rj_dg_3_3",
    "rj_dg_3_4",
    # ring
    "rj_dg_4_1",
    "rj_dg_4_2",
    "rj_dg_4_3",
    "rj_dg_4_4",
    #pinky
    "rj_dg_5_1",
    "rj_dg_5_2",
    "rj_dg_5_3",
    "rj_dg_5_4",
]

ACTUATOR_NAMES = [
    "rj_wrist_0_1_a",
    "rj_wrist_0_2_a",
    "rj_wrist_0_3_a",

    "rj_wrist_1_1_a",
    "rj_wrist_1_2_a",
    "rj_wrist_1_3_a",

    # thumb
    "rj_dg_1_1_a",
    "rj_dg_1_2_a",
    "rj_dg_1_3_a",
    "rj_dg_1_4_a",
    # index
    "rj_dg_2_1_a",
    "rj_dg_2_2_a",
    "rj_dg_2_3_a",
    "rj_dg_2_4_a",
    # middle
    "rj_dg_3_1_a",
    "rj_dg_3_2_a",
    "rj_dg_3_3_a",
    "rj_dg_3_4_a",
    # ring
    "rj_dg_4_1_a",
    "rj_dg_4_2_a",
    "rj_dg_4_3_a",
    "rj_dg_4_4_a",
    #pinky
    "rj_dg_5_1_a",
    "rj_dg_5_2_a",
    "rj_dg_5_3_a",
    "rj_dg_5_4_a",
]

# Table geometry (matches scene_mjx_pick_and_place.xml)
# Body pos="0.32 0.15 0.0375", geom size="0.15 0.15 0.0375"
TABLE_CENTER_X = 0.32
TABLE_CENTER_Y = 0.15
TABLE_HALF_X = 0.15
TABLE_HALF_Y = 0.15
TABLE_SURFACE_Z = 0.075   # body_z + half_z = 0.0375 + 0.0375

CUBE_HALF_SIZE = 0.035
GOAL_Z = TABLE_SURFACE_Z + CUBE_HALF_SIZE  # cube center resting on table = 0.11

# x/y bounds for goal: keep cube fully on table
GOAL_X_MIN = TABLE_CENTER_X - TABLE_HALF_X + CUBE_HALF_SIZE
GOAL_X_MAX = TABLE_CENTER_X + TABLE_HALF_X - CUBE_HALF_SIZE
GOAL_Y_MIN = TABLE_CENTER_Y - TABLE_HALF_Y + CUBE_HALF_SIZE
GOAL_Y_MAX = TABLE_CENTER_Y + TABLE_HALF_Y - CUBE_HALF_SIZE

FINGERTIP_NAMES = [
    "rl_dg_1_tip_c",
    "rl_dg_2_tip_c",
    "rl_dg_3_tip_c",
    "rl_dg_4_tip_c",
    "rl_dg_5_tip_c",
]
