import numpy as np
import h5py

'''
imitation_type = np.dtype([('image', float, (20, 30)), ('label', float, 5)])
print("datatype :\n{}".format(imitation_type))

hdf5_file = h5py.File("carla_dataset_1.hdf5", "a")
dataset = hdf5_file.create_dataset("carla_set1",shape =(1,), maxshape=(None,), chunks=(1,), compression="lzf", dtype=imitation_type)

image = np.ndarray(shape=(20,30), dtype=float)
label = np.array([0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
label2 = np.array([-0.0, -0.25, -0.5, -0.75, -1.0]).astype(float)

data = np.array([(image, label)], dtype=imitation_type)
dataset[0] = data

print(dataset[0,"label"].shape)
print(dataset[0,"image"].shape)

print("label\n", dataset[0, "label"])
print("image\n", dataset[0, "image"])     

dataset.resize(2, axis=0 )
data = np.array([(image, label2)], dtype=imitation_type)
dataset[1] = data

hdf5_file.close()
'''
hdf5_file = h5py.File("carla_dataset_1.hdf5", "r")
print(list(hdf5_file.keys()))
dataset = hdf5_file['carla_set2']

print("label\n", dataset[0, "label"])
print("image\n", dataset[1, "label"])
print("shape ", dataset.shape)