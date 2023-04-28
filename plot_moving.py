import os

import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import cv2

import depth_image_process as MY_DIP
from plot_pointcloud import fake_intrinsics
from depth_image_process import calculate_d_range

intr = fake_intrinsics()
_v = 0.35
_w = 0.2

def calculate_from_d(d,v,w):
    '''
    :param d: float or array, distance from the center of the robot to the target
    :param v: float, linear velocity
    :param w: float, angular velocity
    :return l: float or array, the length the robot would move front on z-axis
    :return t: float or array, the time cost of this move
    '''
    radius = v / w
    cosine = 1 - d/2*radius
    theta = np.arccos(cosine)
    sine = np.sin(theta)
    l = 2 * radius * sine
    t = w / theta
    return l,t

def plot_d_to_other_plots(d_min,dmax):
    d_range = np.linspace(d_min,dmax,100)
    
    l_range,t_range = calculate_from_d(d_range,_v,_w)
    
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