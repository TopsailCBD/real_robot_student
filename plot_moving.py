import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

import depth_image_process as MY_DIP
from depth_image_process import calculate_d_range
from plot_utils import calculate_from_d_smooth, fake_intrinsics

intr = fake_intrinsics()
_v = 0.35
_w = 0.2

def plot_d_to_other_plots(d_min,dmax):
    d_range = np.linspace(d_min,dmax,100)
    
    l_range,t_range = calculate_from_d_smooth(d_range,_v,_w)
    
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