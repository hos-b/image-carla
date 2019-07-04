import numpy as np
import h5py
import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
from utils import action_to_label

DATASET_DIR = "/home/bahadorm"
PKG_NAME = "carla_dataset.hdf5"

print("opening {}".format(PKG_NAME))
data = h5py.File(os.path.join(DATASET_DIR, PKG_NAME), "r")        
keys = list(data.keys())
ds_count = len(keys)

count  = np.zeros((9))

for i in range(ds_count):
    episode = data[keys[i]]
    print("episode {}/{}".format(i,ds_count))
    for j in range(episode.shape[0]) :
        label = action_to_label(episode[j, "label"])
        count[label] +=1

print(count)

# [63405. 10599. 10754. 31964.  5788. 12310.   210.     0.     0.]
# SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -benchmark -fps=20 -carla-server -ResX=640 ResY=480 -windowed