"""evaluation script"""

import argparse
import logging
import random
import time
import os
import numpy as np
import h5py
import subprocess

import torchvision.transforms as transforms
import torch
from utils import label_to_action_dobule
from utils import action_to_label_double
from utils import compare_controls
from utils import print_over_same_line

from server.PythonClient.carla.client import make_carla_client, VehicleControl
from server.PythonClient.carla.sensor import Camera, Lidar
from server.PythonClient.carla.settings import CarlaSettings
from server.PythonClient.carla.tcp import TCPConnectionError
from agent.cagent import CBCAgent

def distance_3d(pose1, pose2):
    return np.sqrt((pose1.x-pose2.x)**2 + (pose1.y-pose2.y)**2 + (pose1.z-pose2.z)**2)

'''
0 - Default
1 - ClearNoon
2 - CloudyNoon
3 - WetNoon
4 - WetCloudyNoon
5 - MidRainyNoon
6 - HardRainNoon
7 - SoftRainNoon
8 - ClearSunset
9 - CloudySunset
10 - WetSunset
11 - WetCloudySunset
12 - MidRainSunset
13 - HardRainSunset
14 - SoftRainSunset
'''
def run_carla_train(total_frames, model, device, history, weather, vehicles, pedestians, DG_next_location, DG_next_episode,DG_threshold) :
    with make_carla_client("localhost", 2000) as client:
        print('carla client connected')
        # setting up HDF5 file
        imitation_type = np.dtype([('image', np.uint8, (512, 512, 3)), ('label', np.float32, 5)])
        hdf5_file = h5py.File(os.path.join("/tmp","dagger_dataset.hdf5"), "a")
        # setting up transform
        transform_list = []
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        # frames trained
        saved_frames = 0
        skipped_frames = 0
        # keeping track of episodes
        dagger_episode_count = 0
        # network input
        network_frames = torch.zeros(1, 3*history, 256, 256).float().to(device)
        for episode in range(10000):
            # dagger end
            if saved_frames >= total_frames:
                break
            # settings to send to carla
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=vehicles,
                NumberOfPedestrians=pedestians,
                WeatherId=weather, 
                QualityLevel='Low') # QualityLevel=args.quality_level
            settings.randomize_seeds()

            camera = Camera('RGBFront', PostProcessing='SceneFinal')
            camera.set_image_size(512, 512)
            camera.set(FOV=120.0)
            camera.set_position(2.0, 0, 1.60)
            camera.set_rotation(roll=0, pitch=-10, yaw=0)
            settings.add_sensor(camera)
            
            # choosing starting position, starting episode
            scene = client.load_settings(settings)
            number_of_player_starts = len(scene.player_start_spots)
            # going through them one by one
            player_start = DG_next_location
            client.start_episode(player_start)
            # keeping track of frames in this episode
            dagger_index = 0
            # record flag, when set the history buffer is filled with the first frame
            # after it is set, all frames should be set
            record = False
            # preventing odd expert behavior
            last_steering = 0
            for frame_index in range(1000):
                measurements, sensor_data = client.read_data()

                # getting expert controls
                control = measurements.player_measurements.autopilot_control
                expert = np.ndarray(shape=(5,), dtype = float)
                control.steer = control.steer if np.abs(control.steer)>5e-4 else 0
                expert[0] = control.steer
                expert[1] = control.throttle
                expert[2] = control.brake
                expert[3] = 1 if control.hand_brake else 0
                expert[4] = 1 if control.reverse else 0
                last_steering = expert[0] if frame_index==0 else 0
                # checking whether the episode should end (i.e. car crash or fucked up stuff)
                # measurements.player_measurements.collision_pedestrians doesn't matter. fuck pedestrians
                if  measurements.player_measurements.collision_vehicles > 0 or \
                    measurements.player_measurements.collision_other > 0 or \
                    measurements.player_measurements.intersection_offroad > 0.15 or \
                    measurements.player_measurements.intersection_otherlane > 0.15 or \
                    measurements.player_measurements.autopilot_control.hand_brake or \
                    measurements.player_measurements.autopilot_control.reverse or \
                    np.abs(last_steering-expert[0])>0.8 or \
                    np.abs(expert[0])==1:
                    break
                last_steering = expert[0]
                # capturing and convering current frame
                dagger_frame = sensor_data['RGBFront'].data
                network_frame = np.transpose(dagger_frame, (1, 0, 2))
                network_frame = transform(network_frame).float().to(device)
                # if this is the first frame, fill the history buffer with the current frame; otherwise shift
                if frame_index==0 :
                    for i in range(history) :
                        network_frames[0, i*3:(i+1)*3] = network_frame
                else :
                    network_frames[0, 3:] = network_frames[0, 0:-3]
                    network_frames[0, :3] = network_frame

                # getting agent predictions
                model.net.eval()
                pred_cls, pred_reg  = model.predict(network_frames)
                pred_cls = torch.argmax(pred_cls)
                pred_cls = pred_cls.item()
                pred_reg = pred_reg.item()
                agent = label_to_action_dobule(pred_cls, pred_reg)
                # sending back agent's controls
                control = VehicleControl()
                control.steer = agent[0]
                # 30km/h speed limit
                control.throttle = agent[1] if measurements.player_measurements.forward_speed * 3.6 <=25 else 0
                control.brake = agent[2]
                control.hand_brake = False
                control.reverse = False
                client.send_control(control)

                # comparing controls (should we save this ?)
                # only save frames after 50, before that it's bullshit
                if not record :
                    if compare_controls(expert=expert[0:3], agent=agent, threshold=DG_threshold) and frame_index > 50:
                        record = True
                        # dataset created only when there are frames to train
                        # it's done here because carla connections tend to fail a lot
                        dataset = hdf5_file.create_dataset("dagger_{:06d}".format(DG_next_episode+dagger_episode_count),shape =(1,), 
                                                            maxshape=(None,), chunks=(1,), compression="lzf", dtype=imitation_type)
                        # increase the index for the next dataset object. done here because 
                        # if it fails mid episode you have to go to the next one ffs fuck carla
                        # it's on the same fucking machine and it's failing to connect
                        dagger_episode_count +=1
                    else :
                        skipped_frames+=1
                else :
                    data = np.array([(dagger_frame, expert)], dtype=imitation_type)
                    dataset.resize(dagger_index+1, axis=0)
                    dataset[dagger_index] = data
                    saved_frames +=1
                    dagger_index += 1
                    if saved_frames>= total_frames:
                        break
                print_over_same_line("dagger frame {}/{} in {} episodes; skipped {}".format(saved_frames,total_frames,dagger_episode_count,skipped_frames))
        hdf5_file.close()
        return dagger_episode_count, skipped_frames-51*dagger_episode_count

def dagger(frames, model, device, history, weather, vehicles, pedestians, DG_next_location, DG_next_episode, DG_threshold):
    while True:
        try:
            episode_count, skipped = run_carla_train(total_frames=frames, model=model, device=device, history=history, 
                                                                              weather=weather, vehicles=vehicles, pedestians=pedestians, 
                                                                              DG_next_episode=DG_next_episode, DG_threshold=DG_threshold,
                                                                              DG_next_location=DG_next_location,)
            print('done')
            return episode_count, skipped
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
