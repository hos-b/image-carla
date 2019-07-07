# Imitation Agent for CARLA Simulator

## Description
The aim is to train an IL agent in the CARLA environment. The internal autopilot is used for data collection.

## Networks
### ResNet18 (~12M Parameters)
This architecture is accessible by using `CBCAgent(..., name='resnet18')`. The output of the convolution layers is passed to an adaptive average pool followed by a fully connected layer. This layer is used to classify the agent's actions :
+ throttle
+ throttle + left
+ throttle + right
+ brake
+ brake + left
+ brake + right
+ no-op
+ left
+ right

### EfficientNet (~5M Parameters)
This architecture is accessible by using `CBCAgent(..., name='efficient-net')`. It produces much better results with less parameters. It is solely used for classification (same 9 classes).

### Custom EfficientNet (~5M Parameters)
This architecture is accessible by using `CBCAgent(..., name='efficient-double')`. The output of the conv layers is passed to two separate fully connected layers. One is used for classification into 3 classes :
+ throttle
+ brake
+ no-op

The second fully connected layer is used to regress the steering angle. A tanh activation function is used to produce an output between -1 and 1. This network produces better results than the base EfficientNet model.

## Algorithms

### Data Collection
Controls from Carla's autopilot is used as the label for each frame. The current dataset is produced from Town02. A total of 110 episodes, each containing ~700 frames have been recorded and stored inside an hdf5 file using `data_collector.py`.

### DataLoader
A custom Dataset class is used to handle the chunks from the hdf5 file. This also makes handling frame history much easier (`dataloader.py`).

### Training
For classification tasks, CrossEntropyLoss is used. In `efficient-double`, MSELoss is used for the steering angle unit. Since the contol distribution is unbalanced, classes are weighed according to their occurance. To get the count for each class, `distribution.py` is ran on the dataset.

### Evaluation
At the end of each epoch, the agent is tested in the simulator for a predefined number of episodes/frames. The following 5 values are used as metrics :
+ average collision with vehicle
+ average collision with pedestrians
+ average collision with other objects
+ average intersection with offroad
+ average intersection with the other lane

### DAgger
At the end of each epoch (before evaluation), the simulator is ran with the current trained agent. The output of the network is compared with that of the expert. If there's a discrepancy, the network is trained with that frame and its corresponding expert controls. Each episode continues until the car crashes or there's a certain intersection with offroad or the other lane. The algorithm itself continues until a predefined number of frames have been trained. 

## Use

### Carla Binaries
This project uses the stable binaries available [here](https://github.com/carla-simulator/carla/releases/tag/0.8.2). The scripts should be placed in the PythonClient subdirectory.

### Args
typical :
+ --snap : save snapshots of the agent
+ --cont : continue training from the latest snapshot
+ -name : name of the run
+ -lr : learning rate
+ -bsize : batch size
+ -epochs : epochs

special :
+ --weighted : use weights for cross entropy loss
+ -history : number of previous frames to append to the input
+ --dagger : use the Dataset Aggregation
+ -dagger_frames : number of frames to train in DAgger
+ -val_episodes : number of validation episodes
+ -val_frames : number of validation frames