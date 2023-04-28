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


def calculate_d_range(intr,shape=(60,108),origin_cv_shape = (848,480), pad=((10,10),(10,18)), depth_thrd=1.5):
    (pad_u, pad_d),(pad_l,pad_r) = pad
    fake_depth = np.zeros(shape)
    fake_depth[pad_u+1:-pad_d,pad_l+1] = depth_thrd * 1000
    fake_depth[pad_u+1:-pad_d,-pad_r-1] = depth_thrd * 1000
    
    resized_fake_depth = cv2.resize(fake_depth, origin_cv_shape, interpolation=cv2.INTER_NEAREST)
    x_range_coordinate = convert_depth_frame_to_pointcloud(resized_fake_depth,intr)
    
    x = x_range_coordinate[:,0]
    return min(x),max(x)

# # 用rs的函数将深度图转换为点云，broken
# def depth_pixel_to_pointcloud(depth_image, intrinsics, depth_pixel):
#     # print(depth_pixel)
#     dis = depth_image[depth_pixel]
    
#     # if dis == 0 or dis > 30000:
#     #     return [0,0,0]
#     camera_coordinate = rs.rs2_deproject_pixel_to_point(intrinsics, depth_pixel, dis)
#     return camera_coordinate

# 用rs的example中的函数（手搓的）将深度图转换为点云
def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics ):
    """
    Convert the depthmap to a 3D point cloud

    Parameters:
    -----------
    depth_frame 	 	 : rs.frame()
                           The depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters

    """
    
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return np.stack([x, y, z]).transpose()

# 用rs的example中的函数（手搓的）将深度图转换为点云，而且传参数
def convert_depth_frame_to_pointcloud_with_args(depth_image, camera_intrinsics, args ):
    """
    Convert the depthmap to a 3D point cloud, save keypoints with in proper range only.

    :param depth_image: array (H,W), depth image, scale:mm.
    :param camera_intrinsics: rs.pyrealsense2.intrinsics, The intrinsic values of the imager in whose coordinate system the depth_frame is computed.
    :param args: dict, argparse in the main file.

    :return coordinates: array (N,3): ([x0,y0,z0],[x1,y1,z1],...)
    - x: array, horizon
    - y: array, height
    - z: array, depth
    """
    
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    
    u, v = np.meshgrid(nx, ny)
    z = depth_image.flatten() / 1000;
    
    z_idx = np.where(args.z_min < z <= args.z_max)
    
    u = u.flatten()[z_idx]
    v = v.flatten()[z_idx]
    z = z[z_idx]
    
    x = (u - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v - camera_intrinsics.ppy)/camera_intrinsics.fy

    y_idx = np.where(args.y_min < y <= args.y_max)
    
    x = x[y_idx]
    y = y[y_idx]
    z = z[y_idx]

    x = np.multiply(x,z)
    y = np.multiply(y,z)

    return np.stack([x, y, z]).transpose()