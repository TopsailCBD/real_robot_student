from a1_robot import Policy
from a1_robot import A1Robot
import time
import numpy as np
import torch
import os

import ctypes
import multiprocessing
import pyrealsense2 as rs
from multiprocessing import Process

from datetime import datetime

from depth_image_process import *

def GetAction(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base):
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    control_finish = np.frombuffer(control_finish_base, dtype=ctypes.c_double)
    reset_finish = np.frombuffer(reset_finish_base, dtype=ctypes.c_double)
    policy = Policy()
    tt = time.time()
    print('begin get action:')
    #obs prompt
    for i in range(10):
        policy.GetAction(obs)
    # policy.trajectory_history[0] = 0
    print('get first action time:', time.time() - tt)


    while(reset_finish[0] <= 0):
        time.sleep(0.00001)

    last_time = time.time()
    last_last_time = last_time
    step = 0
    # states = np.zeros((1000,48))
    # actions = np.zeros((1000, 12))
    while(True):
        while(last_time + 0.02 - time.time() > 0.):
            time.sleep(0.00001)
        print(last_time - last_last_time)
        last_last_time = last_time
        last_time = time.time()
        act[:] = policy.GetAction(obs)
        # states[step] = obs
        # actions[step] = act
        step += 1
        # print('set act to:', act)
        if(control_finish[0] > 0.):
            break
    # tag = datetime.now().strftime('%b%d_%H-%M-%S')
    # path = './data/' + tag
    # os.makedirs(path)
    # np.save(path + '/state.npy', states[:step])
    # np.save(path + '/action.npy', actions[:step])


def RobotControl(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, share_depth_image_base, control_loop_time):
    time.sleep(4)
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)
    control_finish = np.frombuffer(control_finish_base, dtype=ctypes.c_double)
    reset_finish = np.frombuffer(reset_finish_base, dtype=ctypes.c_double)
    robot = A1Robot()

    #reset the robot
    default_motor_angles = robot.default_dof_pos
    current_motor_angles = robot.GetMotorAngles()
    for t in range(2000):
        blend_ratio = min(t / 1000, 1)
        action = blend_ratio * default_motor_angles + (
                1 - blend_ratio) * current_motor_angles
        robot._StepInternal(action,process_action=False)
        time.sleep(0.001)
        
        # ==== Obstacle avoidance ====
        
        if robot.state == 'MARCH':
            # Examine depth image
            # If find obstacle. Set command, start clock
            
        elif robot.state == 'DETOUR_START':
            
        elif robot.state == 'DETOUR_LOOP':
            
        elif robot.state == 'DETOUR_END':
            
        
        obs[:] = robot.ConstructObservation()
        # print('angles:', robot._motor_angles)


    robot._velocity_estimator.reset()
    robot._state_action_counter = 0
    robot._step_counter = 0
    robot._last_reset_time = time.time()
    robot._init_complete = True

    print('reset finish, begin control')
    reset_finish[0] = 1
    #begin control
    begin_control_time = time.time()
    while(True):
        robot._StepInternal(act)
        obs[:] = robot.ConstructObservation()

        control_loop_percent = (time.time() - begin_control_time) / control_loop_time
        # robot.command[0] = 0.3 + 1.6 * min(control_loop_percent, 1/4)

        if(control_loop_percent > 1.0):
            control_finish[0] = 1
            break

    # default_motor_angles = robot.default_dof_pos
    # current_motor_angles = robot.GetMotorAngles()
    # for t in range(2000):
    #     blend_ratio = min(t / 1000, 1)
    #     action = blend_ratio * default_motor_angles + (
    #             1 - blend_ratio) * current_motor_angles
    #     robot._StepInternal(action,process_action=False)
    #     time.sleep(0.001)
    #     obs[:] = robot.ConstructObservation()

def GetDepthImage(shared_depth_image_base, control_finish_base):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)
    depth_image = np.reshape(depth_image, (60, 100))
    control_finish = np.frombuffer(control_finish_base, dtype=ctypes.c_double)

    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # depth_to_disparity = rs.disparity_transform(True)
    # disparity_to_depth = rs.disparity_transform(False)

    # 可以设置参数类似于迭代次数
    decimation.set_option(rs.option.filter_magnitude, 8)

    last_time = time.time()
    last_last_time = last_time

    while(True):
        while(last_time + 0.1 - time.time() > 0.):
            time.sleep(0.001)
        # print('depth image interval:', last_time - last_last_time)
        last_last_time = last_time
        last_time = time.time()
        # Wait for a coherent pair of frames: depth and color
        frame = pipeline.wait_for_frames()

        depth_frame = frame.get_depth_frame()

        depth_frame = decimation.process(depth_frame)
        # depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)


        tmp_depth_image = np.asanyarray(depth_frame.get_data())
        tmp_depth_image = tmp_depth_image / 1000
        tmp_depth_image = tmp_depth_image[:, 8:]
        tmp_depth_image[tmp_depth_image > 5] = 5

        depth_image[:] = (tmp_depth_image - 2.5) / 2.5

        if(control_finish[0] > 0.):
            break

command1 = np.array([0.40, 0, 0.01])
command2 = np.array([0.40, 0, 0.01])
command3 = np.array([0.40, 0, 0.01])
t1,t2,t3 = 0.5,0.5,0.5

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    shared_act_base = multiprocessing.Array(
        ctypes.c_double, 12, lock=False)
    shared_obs_base = multiprocessing.Array(
        ctypes.c_double, 48, lock=False)
    shared_depth_image_base = multiprocessing.Array(
        ctypes.c_double, 60 * 100, lock=False)
    control_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)
    reset_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)


    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    act[:] = np.zeros(12)
    obs[:] = np.zeros(48)

    # Add shared depth image to RobotControl
    p1 = Process(target=RobotControl, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_depth_image_base, 5,))
    p2 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_depth_image_base,))
    p3 = Process(target=GetDepthImage, args=(shared_depth_image_base,control_finish_base))
    p1.start()
    p2.start()
    p3.start()

    os.system("sudo renice -20 -p %s " % (p1.pid))
    os.system("sudo renice -20 -p %s " % (p2.pid))
    os.system("sudo renice -20 -p %s " % (p3.pid))

    p1.join()
    p2.join()
    p3.join()



