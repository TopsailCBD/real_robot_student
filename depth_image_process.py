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


def calculate_d_range(intr,shape=(60,108),origin_cv_shape = (848,480), pad=((10,10),(10,18)), depth_thrd=1.5,polar=False):
    (pad_u, pad_d),(pad_l,pad_r) = pad
    fake_depth = np.zeros(shape)
    fake_depth[pad_u+1:-pad_d,pad_l+1] = depth_thrd * 1000
    fake_depth[pad_u+1:-pad_d,-pad_r-1] = depth_thrd * 1000
    
    resized_fake_depth = cv2.resize(fake_depth, origin_cv_shape, interpolation=cv2.INTER_NEAREST)
    x_range_coordinate = convert_depth_frame_to_pointcloud(resized_fake_depth,intr)
    
    if polar:
        x = x_range_coordinate[:,0]
        z = x_range_coordinate[:,2]
        theta = np.arctan(x/z) # (-pi/2,pi/2)
        return min(theta), max(theta)
    else:
        x = x_range_coordinate[:,0]
        return min(x), max(x)

def calculate_bucket_from_args(intr,args=None):
    '''
    Calculate static parameters of Vector Field from args.
    :param intr: rs.intrinsics()
    :param args: argparse.Namespace()
    :return num_buckets: int, number of buckets
    :return bucket_start: float, d at the 0th bucket
    :return bucket_left_hand: int, shift of index from 0th bucket to d=0
    '''
    assert args is not None
    pad_u,pad_d,pad_l,pad_r = args.pad
    dmin,dmax = calculate_d_range(intr,pad=((pad_u,pad_d),(pad_l,pad_r)),depth_thrd=args.z3,polar=args.polar)
    
    dmin_idx = np.ceil(dmin/args.xscale) # The 0th bucket
    dmax_idx = np.floor(dmax/args.xscale) # The highest bucket
    
    num_buckets = int(dmax_idx - dmin_idx + 1) # Number of buckets, length of vector field
    bucket_start = dmin_idx * args.xscale # d at the 0th bucket
    bucket_left_hand = int(dmin_idx) # Shift of index from 0th bucket to d=0
    
    print('[DIP] Calculate bucket list: num_buckets =',num_buckets,'bucket_start =',bucket_start,'bucket_left_hand =',bucket_left_hand)
    
    return num_buckets,bucket_start,bucket_left_hand

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
def convert_depth_frame_to_pointcloud_with_args(depth_image, camera_intrinsics, scale='mm',args=None ):
    """
    Convert the depthmap to a 3D point cloud, save keypoints with in proper range only.

    :param depth_image: array (H,W), depth image, scale:mm.
    :param camera_intrinsics: rs.pyrealsense2.intrinsics, The intrinsic values of the imager in whose coordinate system the depth_frame is computed.
    :param usez: str, which in z1,z2,z3 to use as threshold
    :param scale: str, scale of depth image, 'mm' or 'm'.
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
    u = u.flatten()
    v = v.flatten()
    
    if scale == 'mm':
        z = depth_image.flatten() / 1000
    else:
        z = depth_image.flatten()
    
    # 直角坐标系下以z为限制
    if not args.polar: 
        z_max = args.z3
        z_idx = np.where((0 < z) & (z <= z_max))
        
        u = u[z_idx]
        v = v[z_idx]
        z = z[z_idx]
    
        x = (u - camera_intrinsics.ppx)/camera_intrinsics.fx
        y = (v - camera_intrinsics.ppy)/camera_intrinsics.fy

        y_idx = np.where((args.ymin < y) & (y <= args.ymax))
        
        x = x[y_idx]
        y = y[y_idx]
        z = z[y_idx]

        x = np.multiply(x,z)
        y = np.multiply(y,z)

        return np.stack([x, y, z]).transpose()
    # 极坐标系下以r为限制
    else:
        x = (u - camera_intrinsics.ppx)/camera_intrinsics.fx
        y = (v - camera_intrinsics.ppy)/camera_intrinsics.fy

        y_idx = np.where((args.ymin < y) & (y <= args.ymax))
        
        x = x[y_idx]
        y = y[y_idx]
        z = z[y_idx]

        x = np.multiply(x,z)
        y = np.multiply(y,z)
        
        r_max = args.z3
        r = np.sqrt(x**2+z**2)
        r_idx = np.where((0 < r) & (r <= r_max))
        
        x = x[r_idx]
        y = y[r_idx]
        z = z[r_idx]
        r = r[r_idx]
        theta  = np.arctan(x/z)
        
        return np.stack([x,y,z,r,theta]).transpose()