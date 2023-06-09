import argparse
import ctypes
import multiprocessing
import os
import time
from datetime import datetime
from multiprocessing import Process
from itertools import product
from functools import partial

import numpy as np
import pyrealsense2 as rs
import torch
import pickle
import cv2

import depth_image_process as MY_DIP
from a1_robot import A1Robot, Policy


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
        # print(last_time - last_last_time)
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


def RobotControl(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_depth_image_base, control_loop_time):
    time.sleep(4)
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)
    depth_image = np.reshape(depth_image,(40,80))
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
        # ==== Obstacle avoidance ====
        # Near the end of the control, freeze the agent.
        if time.time() - begin_control_time > tmax - tn:
            robot.state = 'FROZEN'
            # print('Test finished, freeze the agent')
        
        if no_action or robot.state == 'FROZEN':
            robot._StepInternal(np.zeros(12))
        else:
            robot._StepInternal(act)
        
        if debug:
            # ==== The highest priority: debug ====
            # robot.state = 'DEBUG'
            # robot.command = command_spin
            # print('Command:',robot.command)
            robot.state = 'FROZEN'
            
        elif robot.state is not 'FROZEN':
            # ==== Normal test now ====
            # Examine depth image
            three_part_img = MY_DIP.divide_img(depth_image)
            
            left_obs,mid_obs,right_obs = MY_DIP.ratios_of_three_parts(three_part_img,[side_dist,ctr_dist,side_dist])
            
            left_dea,mid_dea,right_dea = MY_DIP.ratios_of_three_parts(three_part_img,[deadend_dist]*3)
            
            left_col,mid_col,right_col = MY_DIP.ratios_of_three_parts(three_part_img,[collision_dist]*3)
            
            # ==== The first priority: finetuning to avoid collision ====
            # Finetuning last for tf seconds
            if robot.state == 'FINETUNING':
                if time.time() - robot.finetune_clock > tf:
                    print('Finetuning finished, resume origin command')
                    robot.state = old_state
                    robot.command = old_command
                    robot.finetune_clock = -1
                    
            # If obstacles with close distance appears, adjust to the opposite direction
            elif min(left_col,mid_col,right_col) > collision_thrd:
                robot.state = 'FINETUNING'
                # robot.state = 'FROZEN'
                robot.finetune_clock = time.time()
                old_state = robot.state
                old_command = robot.command
                # Avoid collision of the front is in priority
                if max(left_col,right_col) > collision_thrd or mid_col > collision_thrd:
                    print('Detect collision ahead, back off')
                    robot.command = command_back
                    print('Change command to ',robot.command)
                # Avoid collision of left side, turn right
                elif left_col > collision_thrd:
                    print('Detect collision on the left, turn right')
                    robot.command = command_spin * np.array([1,0,-1])
                    print('Change command to ',robot.command)
                # Avoid collision of right side, turn left
                elif right_col > collision_thrd:
                    print('Detect collision on the right, turn left')
                    robot.command = command_spin
                    print('Change command to ',robot.command)
            
            # ==== The second priorty: spinning to explore ==== 
            # If spinning, examine whether can it find a way out
            if robot.state == 'SPINNING':
                # print('I am spinning!',robot.command)
                if min(mid_dea,left_dea,right_dea) < 0.3:
                    robot.state = 'MARCHING'
                    # robot.state = 'FROZEN'
                    robot.command = old_command
                    print('Clear ahead, resume marching')
                    
            # If obstacles with fair distance on all directions, spin
            elif min(mid_dea,left_dea,right_dea) > deadend_thrd:
                robot.state = 'SPINNING'
                # robot.state = 'FROZEN'
                robot.detour_clock = 0
                robot.detour_mode = 0 # Not detouring
                
                robot.command = command_spin
                print('Obstacles around, spinning to find a way out')
                print('Change command to',robot.command)
            
            # ==== The lowest priority: detour to go on the right path ====
            # Other cases, perform detouring as follows
            elif robot.state == 'MARCHING':
                # If find obstacle. Set command, start clock
                if mid_obs > detour_thrd:
                    robot.state = 'DETOUR_START'
                    robot.detour_clock = time.time()
                    
                    if left_obs > right_obs:
                        robot.detour_mode = -1 # Turn right first
                        detour_direction = 'right'
                    else:
                        robot.detour_mode = 1 # Turn left first
                        detour_direction = 'left'
                        
                    print('Detect obstacle, start detour, direction:',detour_direction)
                    
                    old_command = robot.command
                    robot.command = command_deflection * np.array([1,0,robot.detour_mode])
                    print('Change command to',robot.command)
                    
            elif robot.state == 'DETOUR_START':
                if robot.detour_mode != 0 and time.time() - robot.detour_clock > t1:
                    robot.state = 'DETOUR_LOOP'
                    robot.detour_clock = time.time()
                    
                    robot.command = command_detour * np.array([1,0,robot.detour_mode])
                    print('Start step of detour ends.')
                    print('Change command to',robot.command)
                
            elif robot.state == 'DETOUR_LOOP':
                if  robot.detour_mode != 0 and time.time() - robot.detour_clock > t2:
                    robot.state = 'DETOUR_END'
                    robot.detour_clock = time.time()
                    
                    robot.command = command_deflection * np.array([1,0,robot.detour_mode])
                    print('Loop step of detour ends.')
                    print('Change command to',robot.command)
                    
            elif robot.state == 'DETOUR_END':
                if  robot.detour_mode != 0 and time.time() - robot.detour_clock > t3:
                    robot.state = 'MARCHING'
                    robot.detour_clock = -1
                    
                    robot.command = old_command
                    print('End step of detour ends.')
                    print('Change command to',robot.command)
        
        
        # print(robot.state, robot.command)
        
        
        obs[:] = robot.ConstructObservation()

        control_loop_percent = (time.time() - begin_control_time) / control_loop_time
        # robot.command[0] = 0.3 + 1.6 * min(control_loop_percent, 1/4)

        if(control_loop_percent > 1.0):
            control_finish[0] = 1
            break


def GetDepthImage(shared_depth_image_base, control_finish_base):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    # Start streaming
    cfg = pipeline.start(config)
    
    time.sleep(1)
    profile = cfg.get_stream(rs.stream.depth)
    intr = profile.as_video_stream_profile().get_intrinsics()
    print(intr)
    # if save_intr:
    #     pickle.dump(intr, open('../data/intr.pkl', 'wb'))

    depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)    
    depth_image = np.reshape(depth_image, (40, 80))
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
    
    save_flag = False
    if save_step > 0:    
        last_save = last_time
    
    while(True):
        while(last_time + 0.1 - time.time() > 0.):
            time.sleep(0.001)
        # print('depth image interval:', last_time - last_last_time)
        last_last_time = last_time
        last_time = time.time()
        
        if save_step > 0:
            save_flag = False
            if last_time - last_save > save_step:
                last_save = last_time
                save_flag = True
                print('Would save images.',last_time)
        
        # Wait for a coherent pair of frames: depth and color
        frame = pipeline.wait_for_frames()

        depth_frame = frame.get_depth_frame()

        # print(np.asanyarray(depth_frame.get_data()).shape) # (480,848)
        if save_step > 0 and save_flag:
            raw_depth = np.asanyarray(depth_frame.get_data())
            np.save(
                f'../data/raw/{last_time}.npy', 
                raw_depth
            )

        
        depth_frame = decimation.process(depth_frame)
        # depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)


        tmp_depth_image = np.asanyarray(depth_frame.get_data()) # (60,108)
        # print(tmp_depth_image.shape)
        if save_step > 0 and save_flag:
            np.save(
                f'../data/tmp/{last_time}.npy', 
                tmp_depth_image
            )
            
            tmp_depth = tmp_depth[10:-10,10:-18] # (40,80)
            tmp_depth = np.pad(tmp_depth, ((10,10),(10,18)), 'constant', constant_values=0) # (60,108)
            # print(tmp_depth.shape)
            
            resized_depth = cv2.resize(tmp_depth, (848,480), interpolation=cv2.INTER_NEAREST)

            coordinate_list = np.array(list(coordinate_list))
            np.save(
                f'../data/coordinate/{last_time}.npy',
                coordinate_list
            )
                    
        tmp_depth_image = tmp_depth_image / 1000
        tmp_depth_image = tmp_depth_image[:, 8:]
        tmp_depth_image[tmp_depth_image > 5] = 5

        tmp_depth_image = tmp_depth_image[10:-10,10:-10]
        
        if save_step > 0 and save_flag:
            np.save(
                f'../data/final/{last_time}.npy', 
                tmp_depth_image
            )
        
        depth_image[:] = tmp_depth_image[:]
        
        # depth_image[:] = (tmp_depth_image - 2.5) / 2.5

        if(control_finish[0] > 0.):
            break
        
def depth_pixel_to_pointcloud(depth_image, intrinsics, depth_pixel):
    # print(depth_pixel)
    dis = depth_image[depth_pixel]
    
    # if dis == 0 or dis > 30000:
    #     return [0,0,0]
    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrinsics, depth_pixel, dis)
    return camera_coordinate

# ==== Parser Definition ====
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-action', action='store_true')
# parser.add_argument('--save-intr',action='store_true')
parser.add_argument('--save-step',type=float,default=-1)

parser.add_argument('--ctr-dist', type=float, default=1)
parser.add_argument('--side-dist',type=float, default=1.5)
parser.add_argument('--deadend-dist',type=float, default=0.8)
parser.add_argument('--collision-dist',type=float, default=0.4)

parser.add_argument('--detour-thrd',type=float, default=0.3)
parser.add_argument('--deadend-thrd',type=float, default=0.6)
parser.add_argument('--collision-thrd',type=float, default=0.3)

parser.add_argument('-v',type=float,default=0.35)
parser.add_argument('-vb',type=float,default=0.15)

parser.add_argument('-w1',type=float,default=0.24)
parser.add_argument('-w2',type=float,default=0.16)
parser.add_argument('-ws',type=float,default=0.40)
# parser.add_argument('-w3',type=float,default=0.21)

parser.add_argument('-t1',type=float,default=2)
# parser.add_argument('-t2',type=float,default=4)
parser.add_argument('-tf',type=float,default=0.5)
parser.add_argument('-tn',type=float,default=1)

parser.add_argument('-tmax',type=float,default=20)

args = parser.parse_args()

# ==== Parse Global Variables ====
debug = args.debug
no_action = args.no_action
# save_intr = args.save_intr
save_step = args.save_step

ctr_dist = args.ctr_dist
side_dist = args.side_dist
deadend_dist = args.deadend_dist
collision_dist = args.collision_dist

detour_thrd = args.detour_thrd
deadend_thrd = args.deadend_thrd
collision_thrd = args.collision_thrd

command_march = np.array([args.v, 0, 0.01])
command_deflection = np.array([args.v, 0, args.w1])
command_detour = np.array([args.v, 0, -args.w2])
command_spin = np.array([0,0,args.ws])
command_back = np.array([-args.vb,0,0])

# t1,t2,t3 = args.t1, args.t2, args.t1

t1 = args.t1
t2 = 2 * args.w1 * args.t1 / args.w2
t3 = args.t1
tn = args.tn
tf = args.tf
tmax = args.tmax

if __name__ == "__main__":
       
    # ==== Shared Variabes ====
    torch.multiprocessing.set_start_method('spawn')
    shared_act_base = multiprocessing.Array(
        ctypes.c_double, 12, lock=False)
    shared_obs_base = multiprocessing.Array(
        ctypes.c_double, 48, lock=False)
    shared_depth_image_base = multiprocessing.Array(
        ctypes.c_double, 40 * 80, lock=False)
    control_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)
    reset_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)


    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    act[:] = np.zeros(12)
    obs[:] = np.zeros(48)

    # Add shared depth image to RobotControl
    p1 = Process(target=RobotControl, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_depth_image_base, tmax,))
    p2 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, ))
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



