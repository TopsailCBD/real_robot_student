import os
import inspect
import math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import torch

from actor_critic import ActorCritic
from tcn_encoder import TcnEncoder

from a1_robot_velocity_estimator import VelocityEstimator
import numpy as np
import time
import copy



from robot_interface import RobotInterface  # pytype: disable=import-error

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

# load_run = 'length'

load_run = 'v2'

checkpoint = 20000

re_index = [3,4,5,0,1,2,9,10,11,6,7,8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

command = np.array([0.35, 0, 0.01])

# command = np.array([0.35, 0, 0.01])
# command = np.array([0.0, 0, -0.5])
# command = np.array([0.0, 0, 0.5])
# command = np.array([0.35, 0, 0.2])
# command = np.array([0.35, 0, -0.2])

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
    self.trajectory_history = torch.zeros(size=(1, self.history_length, 48), device=self.device)

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
      return raw_action

  #process the raw action of the policy
  def ProcessRawAction(self, action):
    action = action[re_index]
    action = np.clip(action,-action_clip,action_clip)
    return self.default_dof_pos + self._action_scale * action



class A1Robot():
  """Interface for real A1 robot."""
  def __init__(self, **kwargs):

    self._state_action_counter = 0
    self._step_counter = 0

    self._time_step = time_step
    self._action_repeat = action_repeat
    self._action_scale = action_scale
    self.lin_vel_scale = lin_vel_scale
    self.ang_vel_scale = ang_vel_scale
    self.command_scale = command_scale
    self.dof_pos_scale = dof_pos_scale
    self.dof_vel_scale = dof_vel_scale
    self.command = command
    self.gravity_vec = np.array([0, 0, -1])

    # Robot state variables
    self._init_complete = False
    self._base_orientation = None
    self._raw_state = None
    self._last_raw_state = None
    self._motor_angles = np.zeros(12)
    self._motor_velocities = np.zeros(12)
    self._joint_states = None
    self._last_reset_time = time.time()
    self._velocity_estimator = VelocityEstimator(
        self)

    self.default_dof_pos = INIT_MOTOR_ANGLES
    self.last_raw_action = np.zeros(12)
    
    self.state = 'MARCHING'
    self.detour_mode = 0
    self.detour_clock = -1
    self.finetune_clock = -1

    self.device = device



    # Initiate UDP for robot state and actions
    self._robot_interface = RobotInterface()
    for i in range(5):
      self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
      time.sleep(0.002)
      self.ReceiveObservation()






  def ReceiveObservation(self):
    """Receives observation from robot.

    Synchronous ReceiveObservation is not supported in A1,
    so changging it to noop instead.
    """
    state = self._robot_interface.receive_observation()
    self._raw_state = state
    # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
    q = state.imu.quaternion
    self._base_orientation = np.array([q[1], q[2], q[3], q[0]])
    self._motor_angles = np.array([motor.q for motor in state.motorState[:12]])
    self._motor_velocities = np.array(
        [motor.dq for motor in state.motorState[:12]])
    # self._joint_states = np.array(
    #     list(zip(self._motor_angles, self._motor_velocities)))
    if self._init_complete:
      self._velocity_estimator.update(self._raw_state)

  #construct input for policy
  def ConstructObservation(self):
    self.base_ang_vel = self.GetBaseRollPitchYawRate()
    self.dof_pos = self.GetMotorAngles()
    self.dof_vel = self.GetMotorVelocities()
    lin_vel = self.GetBaseVelocity()

    # print('v:', lin_vel)

    self.base_lin_vel = quat_rotate_inverse(self._base_orientation, lin_vel)
    self.projected_gravity = quat_rotate_inverse(self._base_orientation, self.gravity_vec)

    return np.concatenate((self.base_lin_vel * self.lin_vel_scale,
                              self.base_ang_vel * self.ang_vel_scale,
                              self.projected_gravity,
                              self.command * self.command_scale,
                              (self.dof_pos - self.default_dof_pos)[re_index] * self.dof_pos_scale,
                              self.dof_vel[re_index] * self.dof_vel_scale,
                              self.last_raw_action
                              ), axis=-1)




  def ApplyAction(self, motor_commands, control_mode='P'):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    command = np.zeros(60, dtype=np.float32)
    if control_mode == 'P':
      for motor_id in range(num_motors):
        command[motor_id * 5] = motor_commands[motor_id]
        command[motor_id * 5 + 1] = stiffness
        command[motor_id * 5 + 3] = damping
    elif control_mode == 'T':
      for motor_id in range(num_motors):
        command[motor_id * 5 + 4] = motor_commands[motor_id]
    elif control_mode == 'H':
      command = np.array(motor_commands, dtype=np.float32)
    else:
      raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
          control_mode))

    # print(command)

    self._robot_interface.send_command(command)


  def GetBaseRollPitchYawRate(self):
    return np.array(self._raw_state.imu.gyroscope).copy()

  def GetBaseVelocity(self):
    return self._velocity_estimator.estimated_velocity.copy()

  def GetMotorAngles(self):
    return MapToMinusPiToPi(self._motor_angles).copy()

  def GetMotorVelocities(self):
    return self._motor_velocities.copy()

  def GetBaseOrientation(self):
    return self._base_orientation.copy()

  def GetFootContacts(self):
    return np.array(self._raw_state.footForce) > 20

  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
    return analytical_leg_jacobian(motor_angles, leg_id)


  #process the raw action of the policy
  def ProcessRawAction(self, action):
    action = action[re_index]
    action = np.clip(action,-action_clip,action_clip)
    return self.default_dof_pos + self._action_scale * action


  def _StepInternal(self, action = None, process_action = True, control_mode='P'):
    if(process_action):
      self.last_raw_action = action
      action = self.ProcessRawAction(action)
    self.ApplyAction(action, control_mode)
    self.last_processed_action = action
    self.ReceiveObservation()
    self._state_action_counter += 1

  def Terminate(self):
    self._is_alive = False




