import os
import pickle
from functools import partial
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import depth_image_process as MY_DIP
from plot_utils import fake_intrinsics

data_dir = "../data_marching/tmp"
out_dir = "../fig/rectangular"

# 读取深度图
def get_depth_image(file):
    tmp_depth = np.load(os.path.join(data_dir, file)) # (60,108)
    tmp_depth = tmp_depth[10:-10,10:-18] # (40,80)
    tmp_depth = np.pad(tmp_depth, ((10,10),(10,18)), 'constant', constant_values=0) # (60,108)
    # print(tmp_depth.shape)
    
    resized_depth = cv2.resize(tmp_depth, (848,480), interpolation=cv2.INTER_NEAREST)
    # print(resized_depth.shape)
    return resized_depth
    

# 在2D地图上可视化点云，在这里定义grid：直角坐标和极坐标，高度用颜色表示
def plot_point_cloud(coordinates,mode='rectangular',filename=None,img=None):
    """
    将输入的空间坐标序列可视化到xOy平面上

    参数：
    coordinates: numpy数组，形状为(n, 3)，每行包含一个三维坐标点

    返回值：
    无
    """
    fig = plt.figure(figsize=(16,8))
    
    plt.subplot(121)
    plt.imshow(img,cmap='gray')
    
    # print(coordinates[:5])
    selected_coordinates = []
    for i in range(coordinates.shape[0]):
        if -0.3 < coordinates[i,1] < 0.5 and coordinates[i,2] < 1.5:
            selected_coordinates.append(coordinates[i,:])
    selected_coordinates = np.stack(selected_coordinates)
    # print(selected_coordinates.shape)
    
    # 从输入数组中提取x和y坐标
    x = selected_coordinates[:, 0] # / 1000
    y = selected_coordinates[:, 1] # / 1000
    z = selected_coordinates[:, 2] # / 1000

    
    if mode == 'polar':
        r = np.sqrt(x**2+z**2)
        theta = np.arctan2(z,x)        
        
        ax = plt.subplot(122,projection='polar')
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
    elif mode == 'rectangular':
        ax = plt.subplot(122)
        ax.set_aspect(1)
        # 绘制散点图
        # plt.scatter(x, z, c=y,cmap='jet',s=1)
        sca = plt.scatter(x,y,c=z,s=1,vmin=0,vmax=1.5)

        # 设置x和y坐标轴的范围
        x_range = np.max(x) - np.min(x)
        z_range = np.max(z) - np.min(z)
        # plt.xlim([np.min(x) - 0.1 * x_range, np.max(x) + 0.1 * x_range])
        # plt.ylim([np.min(z) - 0.1 * z_range, np.max(z) + 0.1 * z_range])
        
        # plt.xlim(-4.1,0.1)
        # plt.ylim(-0.1,5.1)

        # 添加标题和坐标轴标签
        plt.title('Coordinates in rectangular Plane')
        plt.xlabel('x (Horizon)')
        plt.ylabel('z (Depth)')
        # plt.colorbar()
        
        # 获取颜色条轴
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)

        # 调整颜色条轴的大小
        cbar = plt.colorbar(sca, cax=cax)
        cbar.ax.tick_params(labelsize=8)  # 调整颜色条标签的大小

        plt.grid()
    
    elif mode == '3d':
        ax = plt.subplot(122,projection='3d')

        r = np.sqrt(x**2+z**2)
        ax.scatter(x,z,y,c=r,cmap='jet')
        
        plt.title('Coordinates in 3D')
        # ax.set_xlim(np.min(x),np.max(x))
        # ax.set_ylim(np.min(y),np.max(y))
        # ax.set_zlim(np.min(z),np.max(z))
        ax.set_xlabel('x (Horizon)')
        ax.set_zlabel('y (Height)')
        ax.set_ylabel('z (Depth)')
        plt.grid()
    
    plt.tight_layout()
    if filename is None:
        save_dir = '../fig'
        filename = f'{mode}_test'
    else:
        save_dir = f'../fig/{mode}'
        filename = filename[:-4]
    plt.savefig(f'{save_dir}/{filename}.png',dpi=300)
    plt.close()


if __name__ == '__main__':

    debug = True
    
    intrinsics = fake_intrinsics()
    # intrinsics = pickle.load(open('../intr.pkl','rb'))
    
    file_list = os.listdir(data_dir)
    
    point_idx = product(range(80,400,8),range(80,724,8))

    if debug:
        a_file = file_list[0]
        
        # a_depth = np.load(os.path.join('../data_marching/raw', a_file))
        a_depth = get_depth_image(a_file)
        
        # _depth_pixel_to_pointcloud = partial(depth_pixel_to_pointcloud,a_depth,intrinsics)
        # coordinate_list = map(_depth_pixel_to_pointcloud,point_idx)
        # coordinate_list = np.array(list(coordinate_list))
        
        coordinate_list = MY_DIP.convert_depth_frame_to_pointcloud(a_depth,intrinsics)
        # print(coordinate_list.shape)
        plot_point_cloud(coordinate_list,mode='rectangular',filename=None,img=a_depth)
    else:
        for filename in tqdm(file_list):
            a_depth = get_depth_image(filename)
            
            # _depth_pixel_to_pointcloud = partial(depth_pixel_to_pointcloud,a_depth,intrinsics)
            # coordinate_list = map(_depth_pixel_to_pointcloud,point_idx)
            # coordinate_list = np.array(list(coordinate_list))
            # print(coordinate_list.shape) # (6360,3)
            
            coordinate_list = MY_DIP.convert_depth_frame_to_pointcloud(a_depth,intrinsics)            
            plot_point_cloud(coordinate_list,mode='3d',filename=filename,img=a_depth)