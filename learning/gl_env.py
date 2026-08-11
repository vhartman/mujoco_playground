"""Headless GL and GPU allocator setup. Import before mujoco or jax.

mujoco picks its GL backend at import time, so setting MUJOCO_GL afterwards is
a no-op; PYOPENGL_PLATFORM has to agree with it or mujoco never exposes
Renderer.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# BFC fragments and OOMs when JAX and warp share a GPU.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
