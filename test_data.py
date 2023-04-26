import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data_marching"


def traverse_all_img():
    save = False
    for folder in os.listdir(data_dir):
        if folder.endswith('_img'):
            continue
        for file in os.listdir(os.path.join(data_dir, folder)):
            os.makedirs(os.path.join(data_dir, folder+'_img'), exist_ok=True)
            if file.endswith(".npy"):
                data = np.load(os.path.join(data_dir, folder, file))
                
                if save:
                    if folder in ['raw','tmp']:
                        _vmin,_vmax = 0,3200
                    elif folder in ['final']:
                        _vmin,_vmax = 0,3.2
                        
                    plt.imsave(os.path.join(data_dir, folder+'_img', file[:-4] + ".png"), data, vmin=_vmin, vmax=_vmax, cmap='gray')

def traverse_by_folder():
    file_list = os.listdir(os.path.join(data_dir, 'raw'))
    file_list.sort()
    for file in file_list:
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_dir, 'raw', file))
            data1 = np.load(os.path.join(data_dir, 'tmp', file))
            data2 = np.load(os.path.join(data_dir, 'final', file))
            
            print(file,'raw',data.min(), data.max(),'tmp',data1.min(), data1.max(),'final',data2.min(), data2.max())
            
if __name__ == '__main__':
    traverse_by_folder()