
import numpy as np
import h5py
import os

import torch
import torch.utils.data
import torchvision.transforms


DATASET_DIR = "/home/hosein/part"
PKG_NAME = "carla_dataset.hdf5"


class CarlaHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.data = None
        self.train = kwargs.get("train", True)
        self.history = kwargs.get("history", 1)
        self.validation_episodes = kwargs.get("validation_episodes", 5)
        self.transform  = kwargs.get("transform")
        
        print("opening {}".format(PKG_NAME))
        
        self.keys = list(hdf5_file.keys())
        self.ds_count = len(self.keys)
        
        if self.is_train :
            self.start = 0
            self.end = self.ds_count-self.validation_episodes
        else :
            self.start = self.ds_count-self.validation_episodes
            self.end = self.ds_count

        self.keys = self.keys[start:end]
        self.ds_count = len(self.keys)

        print("found {} episodes".format(self.ds_count))
        self.sizes = np.ndarray(shape=(self.ds_count), type=int)

        for hdf5_index, dataset_index in zip(range(self.start, self.end), range(self.ds_count)):
            self.sizes[dataset_index] = self.hdf5_file[self.keys[hdf5_index]].shape

        self.cummulative_sizes = np.cumsum(self.sizes)
        self.n_samples = self.cummulative_sizes[-1]
        print("total frame count {}".format(running_sum))

    def __getitem__(self, idx):
        if self.data is None :
            self.hdf5_file = h5py.File(os.path.join(DATASET_DIR, PKG_NAME), "r")
        
        episode_key = ''
        frame_index = 0
        last_cs = 0
        for i in range(self.ds_count) :
            if idx < self.cummulative_sizes[i]:
                episode_key = self.keys[i]
                frame_index = idx-last_cs
                break
            last_cs = self.cummulative_sizes[i]
        episode = self.hdf5_file[episode_key]
        
        label = episode[frame_index, "label"]
        samples = torch.zeros(self.history, 3, 640, 480)

        for i in range(self.history):
            history_index = max(0, frame_index-i)
            samples[i] = self.transform(episode[history_index, "image"])

        return label, samples
        
    def __len__(self):
        return self.n_samples


def get_data_loader(batch_size=1, train=False, history=1, validation_episodes=5):

    transform_list = []
    
    #if train:
    #    transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    reader = CarlaHDF5(train=train, history=history, transform=transform, validation_episodes=validation_episodes)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=train,
                                              num_workers=4 if train else 1)
    return data_loader