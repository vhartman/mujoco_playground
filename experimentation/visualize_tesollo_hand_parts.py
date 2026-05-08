import time
import mujoco
import mujoco.viewer
import os

MESH_DIR = (
    "/home/nikola/Projects/DexterousManipulation/mujoco_playground"
    "/mujoco_playground/_src/manipulation/tesollo_hand/xmls/meshes/tesollo"
)

# All 33 STL files (named by stem); used both for the grid and the assembled hand.
MESHES = [
    "rl_dg_mount_c", "rl_dg_mount_c_simple",
    "rl_dg_base_c", "rl_dg_base_c_simple",
    "rl_dg_palm_c",
    "rl_dg_1_1_c", "rl_dg_1_2_c", "rl_dg_1_3_c", "rl_dg_1_4_c",
    "rl_dg_1_tip_c", "rl_dg_1_tip_c_simple",
    "rl_dg_2_1_c", "rl_dg_2_2_c", "rl_dg_2_3_c", "rl_dg_2_4_c",
    "rl_dg_2_tip_c", "rl_dg_2_tip_c_simple",
    "rl_dg_3_1_c", "rl_dg_3_2_c", "rl_dg_3_3_c", "rl_dg_3_4_c", "rl_dg_3_tip_c",
    "rl_dg_4_1_c", "rl_dg_4_2_c", "rl_dg_4_3_c", "rl_dg_4_4_c", "rl_dg_4_tip_c",
    "rl_dg_5_1_c", "rl_dg_5_2_c", "rl_dg_5_3_c", "rl_dg_5_4_c",
    "rl_dg_5_tip_c", "rl_dg_5_tip_c_simple",
]

COLS = 6
SPACING = 0.3

# Assembled hand placed below the grid (y=-0.6).
# Root body in tesollo.xml is pos="-0.05 0 0.03" quat="0 1 0 1"; shift x by +0.7.
HAND_X = 0.65
HAND_Y = -0.6
HAND_Z = 0.03

ASSEMBLED_HAND = f"""
    <!-- ── Assembled hand ── -->
    <body name="rh" pos="{HAND_X} {HAND_Y} {HAND_Z}" quat="0 1 0 1">
      <body name="mount">
        <geom class="visual" type="mesh" mesh="rl_dg_mount_c"/>
        <geom class="visual" pos="0 0 0.004" type="mesh" mesh="rl_dg_base_c"/>
        <geom class="visual" pos="0 0 0.0738" quat="1 0 0 0" type="mesh" mesh="rl_dg_palm_c"/>
        <body name="palm_lower" pos="0 0 0"/>
      </body>
      <!-- Thumb -->
      <body name="rl_dg_1_1" pos="-0.0162 0.019 0.0866">
        <geom class="visual" type="mesh" mesh="rl_dg_1_1_c"/>
        <body name="rl_dg_1_2" pos="0.04195 0 0">
          <geom class="visual" type="mesh" mesh="rl_dg_1_2_c"/>
          <body name="rl_dg_1_3" pos="0 0.031 0">
            <geom class="visual" type="mesh" mesh="rl_dg_1_3_c"/>
            <body name="rl_dg_1_4" pos="0 0.0388 0">
              <geom class="visual" type="mesh" mesh="rl_dg_1_4_c"/>
              <geom class="visual" pos="0 0.0363 0" quat="1 0 0 0" type="mesh" mesh="rl_dg_1_tip_c" material="white"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Pointer -->
      <body name="rl_dg_2_1" pos="-0.0071 0.027 0.1399">
        <geom class="visual" type="mesh" mesh="rl_dg_2_1_c"/>
        <body name="rl_dg_2_2" pos="0.01765 0 0.0265">
          <geom class="visual" type="mesh" mesh="rl_dg_2_2_c"/>
          <body name="rl_dg_2_3" pos="0 0 0.0388">
            <geom class="visual" type="mesh" mesh="rl_dg_2_3_c"/>
            <body name="rl_dg_2_4" pos="0 0 0.0388">
              <geom class="visual" type="mesh" mesh="rl_dg_2_4_c"/>
              <geom class="visual" pos="0 0 0.0255" quat="1 0 0 0" type="mesh" mesh="rl_dg_2_tip_c" material="white"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Middle -->
      <body name="rl_dg_3_1" pos="-0.0071 0.0025 0.1439">
        <geom class="visual" type="mesh" mesh="rl_dg_3_1_c"/>
        <body name="rl_dg_3_2" pos="0.01765 0 0.0265">
          <geom class="visual" type="mesh" mesh="rl_dg_3_2_c"/>
          <body name="rl_dg_3_3" pos="0 0 0.0388">
            <geom class="visual" type="mesh" mesh="rl_dg_3_3_c"/>
            <body name="rl_dg_3_4" pos="0 0 0.0388">
              <geom class="visual" type="mesh" mesh="rl_dg_3_4_c"/>
              <geom class="visual" pos="0 0 0.0255" quat="1 0 0 0" type="mesh" mesh="rl_dg_3_tip_c" material="white"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Ring -->
      <body name="rl_dg_4_1" pos="-0.0071 -0.022 0.1359">
        <geom class="visual" type="mesh" mesh="rl_dg_4_1_c"/>
        <body name="rl_dg_4_2" pos="0.01765 0 0.0265">
          <geom class="visual" type="mesh" mesh="rl_dg_4_2_c"/>
          <body name="rl_dg_4_3" pos="0 0 0.0388">
            <geom class="visual" type="mesh" mesh="rl_dg_4_3_c"/>
            <body name="rl_dg_4_4" pos="0 0 0.0388">
              <geom class="visual" type="mesh" mesh="rl_dg_4_4_c"/>
              <geom class="visual" pos="0 0 0.0255" quat="1 0 0 0" type="mesh" mesh="rl_dg_4_tip_c" material="white"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Little -->
      <body name="rl_dg_5_1" pos="0.0103 -0.0195 0.092">
        <geom class="visual" type="mesh" mesh="rl_dg_5_1_c"/>
        <body name="rl_dg_5_2" pos="0 -0.028 0.0381">
          <geom class="visual" type="mesh" mesh="rl_dg_5_2_c"/>
          <body name="rl_dg_5_3" pos="0 0 0.031">
            <geom class="visual" type="mesh" mesh="rl_dg_5_3_c"/>
            <body name="rl_dg_5_4" pos="0 0 0.0388">
              <geom class="visual" type="mesh" mesh="rl_dg_5_4_c"/>
              <geom class="visual" pos="0 0 0.0363" quat="1 0 0 0" type="mesh" mesh="rl_dg_5_tip_c" material="white"/>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def build_xml():
    assets = "\n".join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in MESHES)

    grid_bodies = []
    for i, m in enumerate(MESHES):
        x = (i % COLS) * SPACING
        y = (i // COLS) * SPACING
        grid_bodies.append(
            f'    <body name="{m}" pos="{x:.2f} {y:.2f} 0">\n'
            f'      <geom type="mesh" mesh="{m}"/>\n'
            f'    </body>'
        )

    return f"""<mujoco>
  <compiler meshdir="{MESH_DIR}"/>

  <!-- Lightbox: camera headlight with strong ambient + 5 directional fills -->
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
  </visual>

  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
  </default>

  <asset>
    <!-- Skybox: blue-grey gradient -->
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.05 0.07 0.1" width="512" height="3072"/>
    <!-- Tiled floor: 2-colour checker -->
    <texture name="checker" type="2d" builtin="checker"
             rgb1="0.3 0.3 0.3" rgb2="0.5 0.5 0.5" width="512" height="512" mark="none"/>
    <material name="floor_mat" texture="checker" texrepeat="4 4" reflectance="0.15"/>
    <material name="black" rgba="0.2 0.2 0.2 1"/>
    <material name="white" rgba="0.9 0.9 0.9 1"/>
{assets}
  </asset>

  <worldbody>
    <!-- Five directional fill lights -->
    <light directional="true" pos="0 0 5" dir="0 0 -1"  diffuse="0.6 0.6 0.6" ambient="0.15 0.15 0.15"/>
    <light directional="true" pos="5 0 3" dir="-1 0 -1" diffuse="0.35 0.35 0.35" ambient="0 0 0"/>
    <light directional="true" pos="-5 0 3" dir="1 0 -1" diffuse="0.35 0.35 0.35" ambient="0 0 0"/>
    <light directional="true" pos="0 5 3" dir="0 -1 -1" diffuse="0.35 0.35 0.35" ambient="0 0 0"/>
    <light directional="true" pos="0 -5 3" dir="0 1 -1" diffuse="0.35 0.35 0.35" ambient="0 0 0"/>

    <!-- Tiled floor (z=-0.01 so meshes rest just above it) -->
    <geom name="floor" type="plane" pos="0.75 0.3 -0.01"
          size="2.5 2.5 0.1" material="floor_mat" contype="0" conaffinity="0"/>

    <!-- ── Individual mesh grid ── -->
{"".join(grid_bodies)}
{ASSEMBLED_HAND}
  </worldbody>
</mujoco>"""


model = mujoco.MjModel.from_xml_string(build_xml())
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY
    while viewer.is_running():
        t = time.time()
        viewer.sync()
        time.sleep(max(0.0, 1 / 60 - (time.time() - t)))
