import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data_frozen"

for folder in os.listdir(data_dir):
    if folder.endswith('_img'):
        continue
    for file in os.listdir(os.path.join(data_dir, folder)):
        os.makedirs(os.path.join(data_dir, folder+'_img'), exist_ok=True)
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_dir, folder, file))
            # print(folder, np.min(data), np.max(data))            
            # print(file,np.count_nonzero(data == np.nan))
            
            if folder in ['raw','tmp']:
                _vmin,_vmax = 0,3200
            elif folder in ['final']:
                _vmin,_vmax = 0,3.2
                
            plt.imsave(os.path.join(data_dir, folder+'_img', file[:-4] + ".png"), data, vmin=_vmin, vmax=_vmax, cmap='gray')