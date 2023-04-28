import os

import numpy as np
import matplotlib.pyplot as plt

def longest_zero_sequence(arr):
    '''
    Output the index (start,end) of the longest zero subsequence
    for u,v in zero sequence: satisfies arr[u:v] == np.zeros(v-u)
    '''
    zero_range = (0,0)
    arr.append(1)
    
    start = -1
    for idx, x in enumerate(arr):
        if x != 0 and start != -1:
            if idx - start > zero_range[1] - zero_range[0]:
                zero_range = (start, idx)
            start = -1
        elif start == -1:
            start = idx
            
    return zero_range


if __name__ == '__main__':
    test_arr = [2,0,0,0,3,0,0,0,0,0,-1,0,0,0,0]
    print(longest_zero_sequence(test_arr))