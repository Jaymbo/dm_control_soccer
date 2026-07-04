# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cartpole_ball domain."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np


_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()
FILE = 'one_joint_ball.xml'


def get_model_and_assets(num_poles=1):
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_path = os.path.join(os.path.dirname(__file__), FILE)
  with open(xml_path, 'r') as f:
    xml_string = f.read()
  # Map the common includes to the actual assets from dm_control
  assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
  return xml_string, assets

@SUITE.add('benchmarking')
def kick(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Kick(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


def _make_model(n_poles):
  """Generates an xml string defining a cart with `n_poles` bodies."""
  xml_string = common.read_model(FILE)
  if n_poles == 1:
    return xml_string
  mjcf = etree.fromstring(xml_string)
  parent = mjcf.find('./worldbody/body/body')  # Find first pole.
  # Make chain of poles.
  for pole_index in range(2, n_poles+1):
    child = etree.Element('body', name='pole_{}'.format(pole_index),
                          pos='0 0 1', childclass='pole')
    etree.SubElement(child, 'joint', name='hinge_{}'.format(pole_index))
    etree.SubElement(child, 'geom', name='pole_{}'.format(pole_index))
    parent.append(child)
    parent = child
  # Move plane down.
  floor = mjcf.find('./worldbody/geom')
  floor.set('pos', '0 0 {}'.format(1 - n_poles - .05))
  # Move cameras back.
  cameras = mjcf.findall('./worldbody/camera')
  cameras[0].set('pos', '0 {} 1'.format(-1 - 2*n_poles))
  cameras[1].set('pos', '0 {} 2'.format(-2*n_poles))
  return etree.tostring(mjcf, pretty_print=True)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cartpole domain."""

  def cart_position(self):
    """Returns the position of the cart."""
    return self.named.data.qpos['slider'][0]

  def angular_vel(self):
    """Returns the angular velocity of the pole."""
    return self.data.qvel[1:]

  def pole_angle_cosine(self):
    """Returns the cosine of the pole angle."""
    return self.named.data.xmat[2:, 'zz']

  def bounded_position(self):
    """Returns the state, with pole angle split into sin/cos."""
    return np.hstack((self.cart_position(),
                      self.named.data.xmat[2:, ['zz', 'xz']].ravel()))


class Kick(base.Task):
  """A Cartpole_ball `Task` to kick the ball.

  State is initialized either close to the target configuration or at a random
  configuration.
  """
  _CART_RANGE = (-.25, .25)
  _ANGLE_COSINE_RANGE = (.995, 1)

  def __init__(self, random=None):
    """Initializes an instance of `Kick`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
    Args:
      physics: An instance of `Physics`.
    """
    physics.named.data.qpos['slider'] = -0.8 + .5*self.random.randn()
    physics.named.data.qpos['knee'] = np.pi + .01*self.random.randn()
    physics.named.data.qvel['slider'] = 0.01 * self.random.randn()
    physics.named.data.qvel['knee'] = 0.01 * self.random.randn()
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns the reward based on the ball's velocity in x-direction."""
    # qvel for a free joint: [vx, vy, vz, wx, wy, wz]
    ball_vel_x = physics.named.data.qvel['ball_joint'][0]
    # print(f'Reward (ball_vel_x): {ball_vel_x}')
    return ball_vel_x