import os

import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import cv2

import depth_image_process as MY_DIP
from plot_pointcloud import fake_intrinsics

intr = fake_intrinsics()
v = 0.35
w = 0.2
radius = v / w

def calculate_d_range(shape=(60,108),origin_cv_shape = (848,480), pad=((10,10),(10,18)), depth_thrd=1.5):
    (pad_u, pad_d),(pad_l,pad_r) = pad
    fake_depth = np.zeros(shape)
    fake_depth[pad_u+1:-pad_d,pad_l+1] = depth_thrd * 1000
    fake_depth[pad_u+1:-pad_d,-pad_r-1] = depth_thrd * 1000
    
    resized_fake_depth = cv2.resize(fake_depth, origin_cv_shape, interpolation=cv2.INTER_NEAREST)
    x_range_coordinate = MY_DIP.convert_depth_frame_to_pointcloud(resized_fake_depth,intr)
    
    x = x_range_coordinate[:,0]
    return min(x),max(x)

def calculate_from_d(d):
    '''
    :param d: float or array, distance from the center of the robot to the target
    :return l: float or array, the length the robot would move front on z-axis
    :return t: float or array, the time cost of this move
    '''
    cosine = 1 - d/2*radius
    theta = np.arccos(cosine)
    sine = np.sin(theta)
    l = 2 * radius * sine
    t = w / theta
    return l,t

def plot_d_to_other_plots(d_min,dmax):
    d_range = np.linspace(d_min,dmax,100)
    
    l_range,t_range = calculate_from_d(d_range)
    
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    for ax,range_,name in zip(axes,[l_range,t_range],['l','t']):
        ax.plot(d_range,range_)
        ax.set_xlabel('d')
        ax.set_ylabel(name)
        ax.grid()
    
    plt.show()
    
    
if __name__ == '__main__':
    dmin,dmax = calculate_d_range()
    plot_d_to_other_plots(dmin,dmax)