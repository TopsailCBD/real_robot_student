import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage as nd
from PIL import Image as im

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

for i in range(100):
    # Wait for a coherent pair of frames: depth and color
    frame = pipeline.wait_for_frames()

    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # 可以设置参数类似于迭代次数
    decimation.set_option(rs.option.filter_magnitude, 8)

    depth_frame = decimation.process(depth_frame)
    # depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    # depth_frame = disparity_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    # for i in range(3):
    #     depth_image = cv2.pyrDown(depth_image)

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image = depth_image / 1000

    depth_image = depth_image[:, 8:]

    print(depth_image.shape)
