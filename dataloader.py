
import numpy as np
import h5py
import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
from utils import action_to_label_double

DATASET_DIR = "/tmp"
PKG_NAME = "carla_dataset.hdf5"
DGR_NAME = "dagger_dataset.hdf5"

class CarlaHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.data = None
        self.train = kwargs.get("train", True)
        self.history = kwargs.get("history", 1)
        self.validation_episodes = kwargs.get("validation_episodes", 5)
        self.transform  = kwargs.get("transform")
        self.hdf5_name = kwargs.get("hdf5_name")
        print("opening {}".format(self.hdf5_name))
        self.data = h5py.File(os.path.join(DATASET_DIR, self.hdf5_name), "r")        
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

        self.cummulative_sizes = np.zeros((self.ds_count))

        print("found {} episodes ({}-{})".format(self.ds_count, start, end))
        self.sizes = np.ndarray(shape=(self.ds_count), dtype=np.uint16)
        runsum = 0
        for index in range(self.ds_count):
            self.sizes[index] = self.data[self.keys[index]].shape[0]
            runsum += self.sizes[index]
            # on the fly fix for broken dataset
            self.cummulative_sizes[index] = runsum -1
        
        # if it's the dagger dataset, revert the fix
        if self.hdf5_name == DGR_NAME :
            self.cummulative_sizes = np.cumsum(self.sizes)

        self.n_samples = self.cummulative_sizes[-1]
        print("total frame count {}".format(self.n_samples))

        self.data.close()
        self.data=None

    def __getitem__(self, idx):
        if self.data is None :
            self.data = h5py.File(os.path.join(DATASET_DIR, self.hdf5_name), "r", libver="latest")
        
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
        
        label, steer = action_to_label_double(episode[frame_index, "label"])
        label = torch.LongTensor([label])
        steer = torch.Tensor([steer])
        samples = torch.zeros(3*self.history, 256, 256).float()

        for i in range(self.history):
            history_index = max(0, frame_index-i)
            np_frame = episode[history_index, "image"]
            samples[i*3:(i+1)*3] = self.transform(np_frame)

        return steer, label, samples
        
    def __len__(self):
        return self.n_samples


def get_data_loader(batch_size=1, train=False, history=1, validation_episodes=5, dagger=False):

    transform_list = []
    transform_list.append(transforms.ToPILImage())
    if train :
        transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    # transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    if not dagger:
        reader = CarlaHDF5(train=train, history=history, transform=transform, validation_episodes=validation_episodes, hdf5_name=PKG_NAME)
    else :
        reader = CarlaHDF5(train=True, history=history, transform=transform, validation_episodes=0, hdf5_name=DGR_NAME)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=train,
                                              num_workers=4 if train else 1)
    return data_loader
