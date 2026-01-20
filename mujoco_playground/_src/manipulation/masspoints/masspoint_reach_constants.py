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

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "masspoints"
SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx_reaching.xml"

NQ = 6
NV = 6
NU = 6

JOINT_NAMES = [
    "mp_0_joint_0",
    "mp_0_joint_1",
    "mp_0_joint_2",

    "mp_1_joint_0",
    "mp_1_joint_1",
    "mp_1_joint_2",
]

ACTUATOR_NAMES = [
    "mp_0_joint_0_a",
    "mp_0_joint_1_a",
    "mp_0_joint_2_a",

    "mp_1_joint_0_a",
    "mp_1_joint_1_a",
    "mp_1_joint_2_a",
]

FINGERTIP_NAMES = [
    "mp_0_tip",
    "mp_1_tip"
]
