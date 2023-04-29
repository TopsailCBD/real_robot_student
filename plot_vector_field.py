import os

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import longest_zero_sequence

if __name__ == '__main__':
    test_arr = [2,0,0,0,3,0,0,0,0,0,-1,0,0,0,0]
    print(longest_zero_sequence(test_arr))