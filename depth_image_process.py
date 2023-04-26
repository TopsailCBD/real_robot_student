import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image as im
from scipy import ndimage as nd

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# config.enable_stream(rs.stream.depth, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

def divide_img(depth_img:np.ndarray,middle_size=None):
    """
    Divide an image to 3 part: left,middle,right
    :param depth_img: np.ndarray
    :param middle_size: Union[int,None]
    :return (np.ndarray*3)
    """
    height,width = depth_img.shape
    if middle_size is None:
        middle_size = height
        
    side_size = (width - middle_size) // 2
    divide1,divide2 = side_size,side_size+middle_size
    
    return depth_img[:,:divide1], depth_img[:,divide1:divide2], depth_img[:,divide2:]

def ratio_of_obstacle(depth_img,depth_thrd):
    """
    Calculate the rate of pixcels that is close enough.
    :param depth_img: np.ndarray of any shape
    :param depth_thrd: int, object close than that would be taken into calculation
    """
    return np.count_nonzero(depth_img < depth_thrd) / depth_img.size

def ratios_of_three_parts(three_img,three_thrd):
    return [ratio_of_obstacle(img,thrd) for img,thrd in zip(three_img,three_thrd)]

