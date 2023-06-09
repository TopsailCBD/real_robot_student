import argparse
import ctypes
import multiprocessing
import os
import pickle
import time
from datetime import datetime
from functools import partial
from itertools import product
from multiprocessing import Process

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

import depth_image_process as MY_DIP
from a1_robot import A1Robot, Policy
from plot_utils import (calculate_from_d, calculate_from_d_smooth,
                        fake_intrinsics, longest_zero_sequence)


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


def RobotControl(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_vector_field_base, control_loop_time):
    global intr,num_buckets,bucket_start,bucket_left_hand
    
    time.sleep(4)
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    # depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)
    # depth_image = np.reshape(depth_image,(40,80))
    vector_field = np.frombuffer(shared_vector_field_base, dtype=ctypes.c_double)
    control_finish = np.frombuffer(control_finish_base, dtype=ctypes.c_double)
    reset_finish = np.frombuffer(reset_finish_base, dtype=ctypes.c_double)
    robot = A1Robot()
    
    # reset the robot
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
            
        vector_collision, vector_close, vector_detour = np.split(vector_field,3)
        
        need_spin_flag = np.count_nonzero(vector_collision[head_left_idx:head_right_idx]) >= num_head_buckets * collision_thrd
        need_detour_flag = np.count_nonzero(vector_close[head_left_idx:head_right_idx]) >= num_head_buckets * collision_thrd
        should_march_flag = np.count_nonzero(vector_detour[head_left_idx:head_right_idx]) < num_head_buckets * collision_thrd
        
        if debug:
            # ==== The highest priority: debug ====
            # robot.state = 'DEBUG'
            # robot.command = command_spin
            # print('Command:',robot.command)
            robot.state = 'FROZEN'
            
        elif robot.state == 'VFH_S':
            if smooth:
                raise NotImplementedError
            else:
                if robot.detour_mode != 0 and not need_spin_flag:
                    robot.state = 'VFH_2'
                    robot.command = command_march
                    robot.detour_clock = time.time()
                    print('Less collision risk, back to VFH detour.')
                    
        elif robot.state == 'VFH_1':
            if smooth:           
                if robot.detour_mode != 0 and time.time() - robot.detour_clock > td:
                    robot.state = 'VFH_3'
                    print('Part I of smooth VFH ends.')
                    
                    robot.detour_clock = time.time()
                    robot.command = command_detour * np.array([1,0,-robot.detour_mode])
                    print('Change command to',robot.command)
            else:
                if robot.detour_mode != 0 and time.time() - robot.detour_clock > ts and should_march_flag:
                    ts = time.time() - robot.detour_clock
                    
                    if ts > tf:
                        robot.state = 'MARCHING'
                        robot.detour_mode = 0
                        print('Harsh VFH got interrupted.')
                    else:
                        robot.state = 'VFH_2'
                        print('Part I of harsh VFH ends.')
                    
                    robot.detour_clock = time.time()
                    robot.command = command_march
                    print('Change command to',robot.command)
                    
        elif robot.state == 'VFH_2':
            if smooth:
                raise NotImplementedError
            else:
                if need_spin_flag:
                    robot.detour_clock = time.time()
                    tm = tm - (time.time() - robot.detour_clock)
                    
                    robot.state = 'VFH_S'
                    head_left_occupied = np.count_nonzero(vector_collision[head_left_idx:head_center_idx])
                    head_right_occupied = np.count_nonzero(vector_collision[head_center_idx:head_right_idx])
                    
                    spinning_direction = 1 if head_left_occupied < head_right_occupied else -1
                    robot.command = command_spin * np.array([1,0,spinning_direction])
                    
                    print('Part II of harsh VFH pauses to avoid collision')
                    print('Change command to',robot.command)
                    
                elif robot.detour_mode != 0 and time.time() - robot.detour_clock > tm:
                    robot.state = 'VFH_3'
                    print('Part II of harsh VFH ends.')
                    
                    robot.detour_clock = time.time()
                    robot.command = command_spin * np.array([1,0,-robot.detour_mode])
                    print('Change command to',robot.command)               
        
        elif robot.state == 'VFH_3':
            if smooth:
                if robot.detour_mode != 0 and time.time() - robot.detour_clock > td:
                    robot.state = 'MARCHING'
                    # robot.state = 'FROZEN'
                    print('Part II of smooth VFH ends.')
                    
                    robot.detour_mode = 0
                    robot.detour_clock = time.time()
                    robot.command = command_march
                    print('Change command to',robot.command)  
            else:
                if robot.detour_mode != 0 and time.time() - robot.detour_clock > ts:
                    robot.state = 'MARCHING'
                    print('Part III of harsh VFH ends.')
                    
                    robot.detour_mode = 0
                    robot.detour_clock = time.time()
                    robot.command = command_march
                    print('Change command to',robot.command)                     
                    
            
        elif robot.state == 'MARCHING':
            # ==== Calculate shift length on horizontal direction ====
            if need_detour_flag:
                zero_start,zero_end = longest_zero_sequence(vector_detour)
                
                if zero_end - zero_start < len(vector_detour) * deadend_thrd:
                    robot.state = 'SPINNING'
                    robot.command = command_spin
                
                else:
                    zero_middle = (zero_start + zero_end) / 2 
                    d = bucket_start + zero_middle * xscale # x when rectangle | theta when polar
                    
                    if d > 0:
                        robot.detour_mode = -1 # Turn right first
                        detour_direction = 'right'
                    else:
                        robot.detour_mode = 1 # Turn left first
                        detour_direction = 'left'
                    
                    robot.state = 'VFH_1'
                    robot.detour_clock = time.time()
                    
                    print(vector_detour)
                    if smooth:
                        td, l = calculate_from_d_smooth(d,args.v,args.w,args.z2,args.polar)
                            
                        print('VFH calculated, start detour, direction:',detour_direction)
                        print('no-obstacle:',(zero_start,zero_end),'d =',d,'l =',l,'td =',td)

                        robot.command = command_detour * np.array([1,0,robot.detour_mode])
                    else:
                        ts, tm = calculate_from_d(d,args.v,args.w,args.z2,args.polar)
                        print('VFH calculated, start detour, direction:',detour_direction)
                        print('no-obstacle:',(zero_start,zero_end),'d =',d,'ts =',ts,'tm =',tm)

                        robot.command = command_spin * np.array([1,0,robot.detour_mode])
                        
                print('Change command to',robot.command)
        
        # print(robot.state,robot.command)
        
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
    
    # time.sleep(1)
    # profile = cfg.get_stream(rs.stream.depth)
    # intr = profile.as_video_stream_profile().get_intrinsics()
    # print('[GetDepthImage] Read intrinsics:',intr)

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
                    
        tmp_depth_image = tmp_depth_image / 1000
        tmp_depth_image[tmp_depth_image > 5] = 5
        tmp_depth_image = tmp_depth_image[pad_u:-pad_d,pad_l:-pad_r]
        
        if save_step > 0 and save_flag:
            np.save(
                f'../data/final/{last_time}.npy', 
                tmp_depth_image
            )
        
        depth_image[:] = tmp_depth_image[:]
        
        # depth_image[:] = (tmp_depth_image - 2.5) / 2.5

        if(control_finish[0] > 0.):
            break
        
def GetVFH(shared_depth_image_base, shared_vector_field_base, control_finish_base):
    global intr,num_buckets,bucket_start,bucket_left_hand
    # Preprocess()
    print('[GetVFH] Get global intrinsics:',intr)
    print('[GetVFH] Get bucket list: num_buckets =',num_buckets,'bucket_start =',bucket_start,'bucket_left_hand =',bucket_left_hand)
    
    # ==== Get depth image ====
    depth_image = np.frombuffer(shared_depth_image_base, dtype=ctypes.c_double)    
    depth_image = np.reshape(depth_image, (40, 80))
    control_finish = np.frombuffer(control_finish_base, dtype=ctypes.c_double)
    vector_field = np.frombuffer(shared_vector_field_base,dtype=ctypes.c_double)

    # ==== Set save clock ====
    last_time = time.time()
    save_flag = False
    if save_step > 0:    
        last_save = last_time
    
    while(True):
        while(last_time + 0.1 - time.time() > 0.):
            time.sleep(0.001)
            
        last_time = time.time()
        
        if save_step > 0:
            save_flag = False
            if last_time - last_save > save_step:
                last_save = last_time
                save_flag = True
                print('Would save coordinates.',last_time)
        
        # ==== Get point cloud from depth image ====
        # print('[GetVFH] From buffer:',depth_image.shape) # (40,80)
        padded_depth = np.pad(depth_image, ((pad_u,pad_d),(pad_l,pad_r)), 'constant', constant_values=0) # (60,108)
        # print('[GetVFH] After padding:',padded_depth.shape) # (60,108)
        resized_depth = cv2.resize(padded_depth, (848,480), interpolation=cv2.INTER_NEAREST) # (480,848)
        # print('[GetVFH] Used in pc:',resized_depth.shape) # (480,848)
        
        coordinate_list = MY_DIP.convert_depth_frame_to_pointcloud_with_args(resized_depth,intr,scale='m',args=args)
        
        if save_step > 0 and save_flag:
            np.save(
                f'../data/coordinate/{last_time}.npy',
                coordinate_list
            )
        # print('[GetVFH] coordinate_list',coordinate_list.shape)
        
        # ==== Get navigate direction from point cloud ====
        bucket_thrd = coordinate_list.shape[0] / num_buckets * occupied_thrd
        
        # dbuckets = np.zeros((num_buckets))
        # for coordinate in coordinate_list:
        #     bucket_idx = int(np.floor(coordinate[0]/xscale) - bucket_left_hand)
        #     dbuckets[bucket_idx] += 1
        
        coordinate_z3 = coordinate_list
        if polar: # [x,y,z,r,theta]
            coordinate_z2 = coordinate_z3[coordinate_z3[:,3] < args.z2]
            coordinate_z1 = coordinate_z2[coordinate_z2[:,3] < args.z1]
        else: # [x,y,z]
            coordinate_z2 = coordinate_z3[coordinate_z3[:,2] < args.z2]
            coordinate_z1 = coordinate_z2[coordinate_z2[:,2] < args.z1]

        
        hierarchical_dbuckets = []
        for coordinates in (coordinate_z1,coordinate_z2,coordinate_z3):
            if polar: # [x,y,z,r,theta]
                dbuckets, bin_edges = np.histogram(
                    coordinates[:,4],
                    bins=num_buckets,
                    range=(bucket_left_hand * xscale, (bucket_left_hand + num_buckets) * xscale)
                )
            else: # [x,y,z]
                dbuckets, bin_edges = np.histogram(
                    coordinates[:,0],
                    bins=num_buckets,
                    range=(bucket_left_hand * xscale, (bucket_left_hand + num_buckets) * xscale)
                )               
            
            dbuckets[dbuckets < bucket_thrd] = 0
            hierarchical_dbuckets.append(dbuckets)
        
        # ==== Publish vector field ====
        vector_field[:] = np.concatenate(hierarchical_dbuckets)[:]
        
        # ==== End control ====    
        if(control_finish[0] > 0.):
            break
    
# ==== Parser Definition ====
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-action', action='store_true')
parser.add_argument('--save-step',type=float,default=-1)

parser.add_argument('--detour-thrd',type=float, default=0.3)
parser.add_argument('--deadend-thrd',type=float, default=0.1)
parser.add_argument('--collision-thrd',type=float, default=0.25)

parser.add_argument('--smooth', action='store_true')
parser.add_argument('--polar', action='store_true')
parser.add_argument('--occupied-thrd',type=float, default=0.5)

parser.add_argument('-pad',type=list,nargs=4,default=[10,10,10,18])

parser.add_argument('-v',type=float,default=0.35)
parser.add_argument('-vb',type=float,default=0.15)
parser.add_argument('-w',type=float,default=0.4)
# parser.add_argument('-w1',type=float,default=0.24)
# parser.add_argument('-w2',type=float,default=0.16)
parser.add_argument('-ws',type=float,default=0.40)
# parser.add_argument('-w3',type=float,default=0.21)

# parser.add_argument('-t1',type=float,default=2)
# parser.add_argument('-t2',type=float,default=4)
parser.add_argument('-tf',type=float,default=5)
parser.add_argument('-tn',type=float,default=1)

parser.add_argument('-head',type=float,default=0.20)
parser.add_argument('-xscale',type=float,default=0.05)
parser.add_argument('-ymin',type=float,default=-0.3)
parser.add_argument('-ymax',type=float,default=0.45)
parser.add_argument('-z1',type=float,default=0.4)
parser.add_argument('-z2',type=float,default=0.6)
parser.add_argument('-z3',type=float,default=1.2)

parser.add_argument('-tmax',type=float,default=20)

args = parser.parse_args()
intr = fake_intrinsics(edition='a1')
num_buckets, bucket_start, bucket_left_hand = MY_DIP.calculate_bucket_from_args(intr,args)
head_left_idx = int(np.floor(-args.head / args.xscale) - bucket_left_hand)
head_right_idx = int(np.floor(args.head / args.xscale) - bucket_left_hand)
head_center_idx = int(np.floor(0 / args.xscale) - bucket_left_hand)
num_head_buckets = head_right_idx - head_left_idx

# ==== Parse Global Variables ====
debug = args.debug
no_action = args.no_action
save_step = args.save_step

detour_thrd = args.detour_thrd
deadend_thrd = args.deadend_thrd
collision_thrd = args.collision_thrd

smooth = args.smooth
polar = args.polar
occupied_thrd = args.occupied_thrd

pad_u,pad_d,pad_l,pad_r = args.pad

command_march = np.array([args.v, 0, 0.01])
command_detour = np.array([args.v, 0, args.w])
command_spin = np.array([0,0,args.ws])
command_back = np.array([-args.vb,0,0])

xscale = args.xscale

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
    shared_vector_field_base = multiprocessing.Array(
        ctypes.c_double, 3 * num_buckets, lock=False)
    control_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)
    reset_finish_base = multiprocessing.Array(
        ctypes.c_double, 1, lock=False)


    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_double)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_double)
    act[:] = np.zeros(12)
    obs[:] = np.zeros(48)
    
    # Add shared depth image to RobotControl
    p1 = Process(target=RobotControl, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, shared_vector_field_base, tmax,))
    p2 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, control_finish_base,reset_finish_base, ))
    p3 = Process(target=GetDepthImage, args=(shared_depth_image_base,control_finish_base))
    p4 = Process(target=GetVFH, args=(shared_depth_image_base,shared_vector_field_base, control_finish_base))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    os.system("sudo renice -20 -p %s " % (p1.pid))
    os.system("sudo renice -20 -p %s " % (p2.pid))
    os.system("sudo renice -20 -p %s " % (p3.pid))
    os.system("sudo renice -20 -p %s " % (p4.pid))
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()


