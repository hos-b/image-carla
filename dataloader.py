
import numpy as np
import h5py
import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
from utils import action_to_label

DATASET_DIR = "/tmp"
PKG_NAME = "carla_dataset.hdf5"

class CarlaHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.data = None
        self.train = kwargs.get("train", True)
        self.history = kwargs.get("history", 1)
        self.validation_episodes = kwargs.get("validation_episodes", 5)
        self.transform  = kwargs.get("transform")

        print("opening {}".format(PKG_NAME))
        self.data = h5py.File(os.path.join(DATASET_DIR, PKG_NAME), "r")        
        self.keys = list(self.data.keys())
        self.ds_count = len(self.keys)
        
        if self.train :
            start = 0
            end = self.ds_count-self.validation_episodes
        else :
            start = self.ds_count-self.validation_episodes
            end = self.ds_count

        self.keys = self.keys[start:end]
        self.ds_count = len(self.keys)

        print("found {} episodes ({}-{})".format(self.ds_count, start, end))
        self.sizes = np.ndarray(shape=(self.ds_count), dtype=np.uint16)
        for index in range(self.ds_count):
            self.sizes[index] = self.data[self.keys[index]].shape[0]

        self.cummulative_sizes = np.cumsum(self.sizes)
        self.n_samples = self.cummulative_sizes[-1]
        print("total frame count {}".format(self.n_samples))

        self.data.close()
        self.data=None

    def __getitem__(self, idx):
        if self.data is None :
            self.data = h5py.File(os.path.join(DATASET_DIR, PKG_NAME), "r")
        
        episode_key = ''
        frame_index = 0
        last_cumsum = 0
        for i in range(self.ds_count) :
            if idx < self.cummulative_sizes[i]:
                episode_key = self.keys[i]
                frame_index = idx-last_cumsum
                break
            last_cumsum = self.cummulative_sizes[i]
        episode = self.data[episode_key]
        
        label = action_to_label(episode[frame_index, "label"])
        label = torch.LongTensor([label])
        samples = torch.zeros(3*self.history, 512, 512).float()

        for i in range(self.history):
            history_index = max(0, frame_index-i)
            np_frame = episode[history_index, "image"]
            samples[i*3:(i+1)*3] = self.transform(np_frame)

        return label, samples
        
    def __len__(self):
        return self.n_samples


def get_data_loader(batch_size=1, train=False, history=1, validation_episodes=5):

    transform_list = []
    
    #if train:
    transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    transform_list.append(transforms.ToPILImage())
    # transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform_list.append(transforms.Resize(256))
    transform = transforms.Compose(transform_list)

    reader = CarlaHDF5(train=train, history=history, transform=transform, validation_episodes=validation_episodes)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=train,
                                              num_workers=4 if train else 1)
    return data_loader