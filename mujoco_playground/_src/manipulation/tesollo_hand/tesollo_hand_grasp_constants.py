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
SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx_grasping.xml"

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
    "wrist_tx",
    "wrist_ty",
    "wrist_tz",

    "wrist_rx",
    "wrist_ry",
    "wrist_rz",

    # thumb
    "dg_1_j1_rx",
    "dg_1_j2_rz",
    "dg_1_j3_rx",
    "dg_1_j4_rx",
    # index
    "dg_2_j1_rx",
    "dg_2_j2_ry",
    "dg_2_j3_ry",
    "dg_2_j4_ry",
    # middle
    "dg_3_j1_rx",
    "dg_3_j2_ry",
    "dg_3_j3_ry",
    "dg_3_j4_ry",
    # ring
    "dg_4_j1_rx",
    "dg_4_j2_ry",
    "dg_4_j3_ry",
    "dg_4_j4_ry",
    #pinky
    "dg_5_j1_rz",
    "dg_5_j2_rx",
    "dg_5_j3_ry",
    "dg_5_j4_ry",
]

FINGERTIP_NAMES = [
    "rl_dg_1_tip_c",
    "rl_dg_2_tip_c",
    "rl_dg_3_tip_c",
    "rl_dg_4_tip_c",
    "rl_dg_5_tip_c",
]
