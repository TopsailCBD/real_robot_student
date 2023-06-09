import numpy as np
import pyrealsense2 as rs

# Never import matplotlib here.

def fake_intrinsics(edition='default'):
    # [ 848x480  p[424.337 239.504]  f[421.504 421.504]  Brown Conrady [0 0 0 0 0] ]
    if edition == 'default':
        intrinsics = rs.pyrealsense2.intrinsics()
    elif edition == 'a1':
        intrinsics = rs.intrinsics()
    intrinsics.width = 848
    intrinsics.height = 480
    intrinsics.ppx = 424.337
    intrinsics.ppy = 239.504
    intrinsics.fx = 421.504
    intrinsics.fy = 421.504
    intrinsics.model = rs.distortion.brown_conrady
    print(intrinsics)
    return intrinsics

def calculate_from_d_smooth(d,v,w,z=None,polar=False):
    '''
    :param d: float or array, distance from the center of the robot to the target
    :param v: float, linear velocity
    :param w: float, angular velocity
    :return l: float or array, the length the robot would move front on z-axis
    :return t: float or array, the time cost of this move
    '''
    if polar:
        d = z * np.sin(d) # d is theta
    
    radius = v / w
    cosine = 1 - np.abs(d)/2*radius
    phi = np.arccos(cosine)
    sine = np.sin(phi)
    l = 2 * radius * sine
    t = phi / w
    return t, l

def calculate_from_d(d,v,w,z,polar=None):
    '''
    :param d: float or array, distance from the center of the robot to the target
    :param v: float, linear velocity
    :param w: float, angular velocity
    :param z: fl
    :return ts: float or array, the time cost of spinning move
    :return tm, float or array, the time cost of marching move
    '''
    if polar:
        theta = np.abs(d)
    else:
        tangent = np.abs(d) / z
        theta = np.arctan(tangent)
    ts = theta / w
    tm = np.sqrt(d**2 + z**2) / v
    return ts,tm
        
def longest_zero_sequence(arr):
    '''
    Output the index (start,end) of the longest zero subsequence
    for u,v in zero sequence: satisfies arr[u:v] == np.zeros(v-u)
    :param arr: list or np.ndarray
    '''
    zero_range = (0,0)
    if type(arr) is list:
        arr.append(1)
    elif type(arr) is np.ndarray:
        arr = np.append(arr,1)
    
    start = -1
    for idx, x in enumerate(arr):
        if x != 0 and start != -1:
            if idx - start > zero_range[1] - zero_range[0]:
                zero_range = (start, idx)
            start = -1
        elif start == -1:
            start = idx
            
    return zero_range
