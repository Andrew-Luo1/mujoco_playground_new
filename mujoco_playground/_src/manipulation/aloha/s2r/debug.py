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
"""Debug environment for domain randomization."""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha.base import get_assets
from mujoco_playground._src.mjx_env import State


def default_vision_config() -> config_dict.ConfigDict:
  """Default vision configuration for the environment."""
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=32,
      render_height=32,
      enabled_geom_groups=[1, 2, 5],
      use_rasterizer=False,
      enabled_cameras=[4, 5],  # Left and right gripper cameras
  )


class DebugDomainRandomize(mjx_env.MjxEnv):
  """Debug environment for domain randomization."""

  def __init__(self, vision: bool = True):
    """Initialize the environment."""
    # Create default config
    config = config_dict.ConfigDict()
    config.ctrl_dt = 0.02
    config.sim_dt = 0.002

    # Call parent constructor with default config
    super().__init__(config)

    # Initialize environment-specific attributes
    self._xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "aloha"
        / "xmls"
        / "s2r"
        / "mjx_peg_insertion.xml"
    )
    # Load models
    self._mj_model = mujoco.MjModel.from_xml_path(
        self._xml_path.as_posix(), assets=get_assets()
    )
    self._mjx_model = mjx.put_model(self._mj_model)
    self._init_q = self._mj_model.keyframe("home").qpos
    self._init_ctrl = self._mj_model.keyframe("home").ctrl
    # Define action space size
    self._action_size = self._mj_model.nu

    # Initialize renderer with default vision config
    vision_config = default_vision_config()
    if vision:
      from madrona_mjx.renderer import BatchRenderer

      render_height = vision_config.render_height
      render_width = vision_config.render_width

      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=vision_config.gpu_id,
          num_worlds=vision_config.render_batch_size,
          batch_render_view_height=render_height,
          batch_render_view_width=render_width,
          enabled_geom_groups=np.asarray(vision_config.enabled_geom_groups),
          enabled_cameras=np.asarray(vision_config.enabled_cameras),
          add_cam_debug_geo=False,
          use_rasterizer=vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state.

    Args:
      rng: Random number generator key.

    Returns:
      Initial state.
    """
    # Use the RNG or tracing fails and batched rendering breaks.
    rng, rng_reset = jax.random.split(rng)
    _init_q = jnp.array(self._init_q)
    _init_q = _init_q.at[0].set(
        _init_q[0] + jax.random.uniform(rng_reset, (), minval=0.1, maxval=0.1)
    )
    # Initialize MJX data
    data = mjx_env.init(
        self.mjx_model,
        jnp.array(_init_q),
        jnp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=jnp.array(self._init_ctrl),
    )

    # Initialize renderer and get initial RGB and depth images
    render_token, rgb, depth = self.renderer.init(data, self._mjx_model)

    # Create observation with RGB images for left and right grippers
    obs = {
        "pixels/view_0": (
            jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
        ),  # Left gripper view
        "pixels/view_1": (
            jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
        ),  # Right gripper view
    }
    reward, done = jnp.zeros(2)
    # Create initial state
    state = State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics={},
        info={"render_token": render_token},  # Store render token for step
    )

    return state

  def step(self, state: State, action: jax.Array) -> State:
    raise NotImplementedError(
        "Stepping is not implemented for this environment."
    )

  @property
  def xml_path(self) -> str:
    """Path to the xml file for the environment."""
    return self._xml_path

  @property
  def action_size(self) -> int:
    """Size of the action space."""
    return self._action_size

  @property
  def mj_model(self) -> mujoco.MjModel:
    """Mujoco model for the environment."""
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    """Mjx model for the environment."""
    return self._mjx_model


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Apply domain randomization to camera positions, lights, and materials."""
  mj_model = DebugDomainRandomize(vision=False).mj_model
  table_geom_id = mj_model.geom("table").id
  peg_geom_id = mj_model.geom("red_peg").id

  @jax.vmap
  def rand(rng):
    geom_rgba = model.geom_rgba

    # Perturb the peg's off-colors
    pert_colors = jax.random.uniform(rng, (2,), minval=0.0, maxval=1.0)
    geom_rgba = geom_rgba.at[peg_geom_id, 1:3].set(pert_colors)

    # Next, set the floor to a random gray-ish color.
    floor_rgba = geom_rgba[table_geom_id]
    floor_rgba = floor_rgba.at[0].set(
        jax.random.uniform(rng, (), minval=0.0, maxval=1.0)
    )
    floor_rgba = floor_rgba.at[1].set(floor_rgba[0])
    floor_rgba = floor_rgba.at[2].set(floor_rgba[0])
    geom_rgba = geom_rgba.at[table_geom_id].set(floor_rgba)

    return geom_rgba

  geom_rgba = rand(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_rgba": 0,
  })

  model = model.tree_replace({
      "geom_rgba": geom_rgba,
  })

  return model, in_axes
