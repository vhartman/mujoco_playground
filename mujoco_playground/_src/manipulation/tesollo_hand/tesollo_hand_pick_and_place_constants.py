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
"""Constants for tesollo hand."""

import dataclasses

import mujoco

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

FINGERTIP_NAMES = [
    "rl_dg_1_tip_c",
    "rl_dg_2_tip_c",
    "rl_dg_3_tip_c",
    "rl_dg_4_tip_c",
    "rl_dg_5_tip_c",
]


@dataclasses.dataclass(frozen=True)
class SceneGeometry:
    """Table and cube geometry read directly from the MuJoCo model.

    The XML is the single source of truth; this dataclass derives everything
    from it so that Python and XML can never disagree.
    """

    table_center_x: float
    table_center_y: float
    table_half_x: float
    table_half_y: float
    table_surface_z: float
    cube_half_size: float

    @property
    def goal_z(self) -> float:
        return self.table_surface_z + self.cube_half_size

    @property
    def goal_x_min(self) -> float:
        return self.table_center_x - self.table_half_x + self.cube_half_size

    @property
    def goal_x_max(self) -> float:
        return self.table_center_x + self.table_half_x - self.cube_half_size

    @property
    def goal_y_min(self) -> float:
        return self.table_center_y - self.table_half_y + self.cube_half_size

    @property
    def goal_y_max(self) -> float:
        return self.table_center_y + self.table_half_y - self.cube_half_size

    @classmethod
    def from_mj_model(cls, mj_model: mujoco.MjModel) -> "SceneGeometry":
        table_pos  = mj_model.body("table").pos
        table_size = mj_model.geom("table_top").size
        cube_size  = mj_model.geom("cube").size
        return cls(
            table_center_x  = float(table_pos[0]),
            table_center_y  = float(table_pos[1]),
            table_half_x    = float(table_size[0]),
            table_half_y    = float(table_size[1]),
            table_surface_z = float(table_pos[2] + table_size[2]),
            cube_half_size  = float(cube_size[0]),
        )
