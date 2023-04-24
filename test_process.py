# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
# pytype: disable=attribute-error
"""Real robot interface of A1 robot."""

import os
import inspect
import math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import torch

from actor_critic import ActorCritic
from tcn_encoder import TcnEncoder

import numpy as np
import time
import copy


#right first
INIT_MOTOR_ANGLES = np.array([-0.1, 0.8, -1.5, 0.1,0.8,-1.5, -0.1,1.0,-1.5,0.1,1.0,-1.5])
TWO_PI = 2 * math.pi

num_motors = 12
control_mode = 'P'
stiffness = 55  # [N*m/rad]
damping = 0.8  # [N*m*s/rad]
# action scale: target angle = actionScale * action + defaultAngle
action_scale = 0.25
action_clip = 4
action_repeat = 8
time_step = 0.02
lin_vel_scale = 2.0
ang_vel_scale = 0.25
command_scale = np.array([2.0,2.0,0.25])
dof_pos_scale = 1.0
dof_vel_scale = 0.05

history_length = 50
num_obs = 60
num_actions = 12
# load_run = 'action_magnitude/0.01/1'
load_run = 'student/1'
checkpoint = 10000

re_index = [3,4,5,0,1,2,9,10,11,6,7,8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MapToMinusPiToPi(angles):
  """Maps a list of angles to [-pi, pi].

  Args:
    angles: A list of angles in rad.

  Returns:
    A list of angle mapped to [-pi, pi].
  """
  mapped_angles = copy.deepcopy(angles)
  for i in range(len(angles)):
    mapped_angles[i] = math.fmod(angles[i], TWO_PI)
    if mapped_angles[i] >= math.pi:
      mapped_angles[i] -= TWO_PI
    elif mapped_angles[i] < -math.pi:
      mapped_angles[i] += TWO_PI
  return mapped_angles


def analytical_leg_jacobian(leg_angles, leg_id):
  """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a left (1) or right(-1) leg.
  """
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * (-1)**(leg_id + 1)

  t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
  l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
  t_eff = t2 + t3 / 2
  J = np.zeros((3, 3))
  J[0, 0] = 0
  J[0, 1] = -l_eff * np.cos(t_eff)
  J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
      t_eff) / 2
  J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
  J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
  J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
      t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
  J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
  J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
  J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
      t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
  return J

#q shape([4,], v shape([3,])
def quat_rotate_inverse(q, v):
  q_w = q[-1]
  q_vec = q[:3]
  a = v * (2.0 * q_w ** 2 - 1.0)
  b = np.cross(q_vec, v) * q_w * 2.0
  c = q_vec * np.dot(q_vec, v) * 2.0
  return a - b + c

def get_load_path(root, load_run='raw', checkpoint=-1):
  load_run = os.path.join(root, load_run)
  model = "model_{}.pt".format(checkpoint)

  load_path = os.path.join(load_run, model)
  return load_path





class Policy():
  def __init__(self, **kwargs):
    self.device = device
    self.history_length = history_length
    self.policy = self.BuildPolicy()
    self.default_dof_pos = INIT_MOTOR_ANGLES
    self._action_scale = action_scale

  def BuildPolicy(self):
    self.actor_critic = ActorCritic(num_actor_obs=num_obs, num_critic_obs=num_obs, num_actions=num_actions)
    self.student_encoder = TcnEncoder(history_length=self.history_length, device=self.device).to(self.device)

    # load pretrained policy
    resume_path = get_load_path('./pretrained_model', load_run=load_run, checkpoint=checkpoint)
    print(f"Loading model from: {resume_path}")
    loaded_dict = torch.load(resume_path, map_location=self.device)
    self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    self.student_encoder.load_state_dict(loaded_dict['encoder_dict'], strict=False)
    self.trajectory_history = torch.zeros(size=(1, self.history_length, 48), dtype=torch.float32,device=self.device)

    self.actor_critic.eval()
    self.actor_critic.to(self.device)
    return self.actor_critic.act_inference

  def GetAction(self,obs):
    obs = torch.FloatTensor(obs).to(self.device)
    obs = obs.unsqueeze(0)
    self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)
    with torch.inference_mode():
      self.student_z = self.student_encoder(self.trajectory_history)
      concat_obs = torch.concat((obs, self.student_z), dim=-1)

      raw_action = self.policy(concat_obs).cpu().numpy()[0]
      # processed_action = self.ProcessRawAction(raw_action)
      # self.action = processed_action
      return raw_action

  #process the raw action of the policy
  def ProcessRawAction(self, action):
    action = action[re_index]
    action = np.clip(action,-action_clip,action_clip)
    return self.default_dof_pos + self._action_scale * action





policy = Policy()
obs = np.random.random((48))
act = policy.GetAction(obs)
print(act)
policy.trajectory_history[0] = 0


