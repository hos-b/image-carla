import numpy as np
import h5py

        data = h5py.File("tmp/carla_dataset.hdf5", "r")        
        keys = list(data.keys())
        ds_count = len(keys)

        for index in range(ds_count):
            print("{} : {} frames".format(keys[index] ,data[keys[index]].shape[0]))
            runsum += sizes[index]
            
        data.close()
        data=None
