import os
from functools import partial
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm

data_dir = "../data_marching/tmp"
out_dir = "../fig/rectangular"

# 读取深度图
def get_depth_image(file):
    tmp_depth = np.load(os.path.join(data_dir, file))
    # print(tmp_depth.shape)
    resized_depth = cv2.resize(tmp_depth, (848,480), interpolation=cv2.INTER_NEAREST)
    # print(resized_depth.shape)
    return resized_depth
    
# 将深度图转换为点云
def depth_pixel_to_pointcloud(depth_image, intrinsics, depth_pixel):
    # print(depth_pixel)
    dis = depth_image[depth_pixel]
    
    if dis == 0 or dis > 30000:
        return [0,0,0]
    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrinsics, depth_pixel, dis)
    return camera_coordinate

# 在2D地图上可视化点云，在这里定义grid：直角坐标和极坐标，高度用颜色表示
def plot_point_cloud(coordinates,polar=False,filename=None):
    """
    将输入的空间坐标序列可视化到xOy平面上

    参数：
    coordinates: numpy数组，形状为(n, 3)，每行包含一个三维坐标点

    返回值：
    无
    """
    fig = plt.figure()
    
    # 从输入数组中提取x和y坐标
    x = coordinates[:, 0] / 1000
    y = coordinates[:, 1] / 1000
    z = coordinates[:, 2] / 1000

    
    if polar:
        r = np.sqrt(x**2+z**2)
        theta = np.arctan2(z,x)        
        
        plt.subplot(111,projection='polar')
        plt.scatter(theta,r,c=y,cmap='jet',s=1,marker='D',vmin=-2,vmax=3)
        
        # x_range = np.max(r) - np.min(r)
        # y_range = np.max(theta) - np.min(theta)
        # plt.xlim([np.min(r) - 0.1 * x_range, np.max(r) + 0.1 * x_range])
        # plt.ylim([np.min(theta) - 0.1 * y_range, np.max(theta) + 0.1 * y_range])
        
        # plt.ylim(np.pi,2*np.pi)
        
        plt.title('Coordinates in polar Plane')
        plt.xlabel('r (Distance)')
        plt.ylabel('theta (Rotation)')
        plt.colorbar()
        plt.grid()
    else:
        plt.subplot(111)
        # 绘制散点图
        plt.scatter(x, z, c=y,cmap='jet',s=1,marker='D',vmin=-2,vmax=3)

        # 设置x和y坐标轴的范围
        x_range = np.max(x) - np.min(x)
        z_range = np.max(z) - np.min(z)
        # plt.xlim([np.min(x) - 0.1 * x_range, np.max(x) + 0.1 * x_range])
        # plt.ylim([np.min(z) - 0.1 * z_range, np.max(z) + 0.1 * z_range])
        
        plt.xlim(-4.1,0.1)
        plt.ylim(-0.1,5.1)

        # 添加标题和坐标轴标签
        plt.title('Coordinates in rectangular Plane')
        plt.xlabel('x (Horizon)')
        plt.ylabel('z (Depth)')
        plt.colorbar()
        plt.grid()
    
    plt.tight_layout()
    if filename is None:
        save_dir = '../fig'
        filename = 'polar_test' if polar else 'rectangular_test'
    else:
        save_dir = '../fig/polar' if polar else '../fig/rectangular'
        filename = filename[:-4]
    plt.savefig(f'{save_dir}/{filename}.png',dpi=300)
    plt.close()

def fake_intrinsics():
    # [ 848x480  p[424.337 239.504]  f[421.504 421.504]  Brown Conrady [0 0 0 0 0] ]
    intrinsics = rs.pyrealsense2.intrinsics()
    intrinsics.width = 848
    intrinsics.height = 480
    intrinsics.ppx = 424.337
    intrinsics.ppy = 239.504
    intrinsics.fx = 421.504
    intrinsics.fy = 421.504
    intrinsics.model = rs.distortion.brown_conrady
    print(intrinsics)
    return intrinsics

if __name__ == '__main__':
    intrinsics = fake_intrinsics()
    
    file_list = os.listdir(data_dir)
    
    # a_file = file_list[0]
    # a_depth = get_depth_image(a_file)
    
    for filename in tqdm(file_list):
        a_depth = get_depth_image(filename)
        _depth_pixel_to_pointcloud = partial(depth_pixel_to_pointcloud,a_depth,intrinsics)
        
        # for i, j in product(range(60),range(108)):
        #     a_coordinate = _depth_pixel_to_pointcloud((i,j))
        # print(a_coordinate)
        
        coordinate_list = map(_depth_pixel_to_pointcloud,product(range(80,400,8),range(80,724,8)))
        coordinate_list = np.array(list(coordinate_list))
        # print(coordinate_list.shape) # (6360,3)
        
        plot_point_cloud(coordinate_list,polar=False,filename=filename,img=a_depth)