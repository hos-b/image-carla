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
from utils import compare_controls
from utils import print_over_same_line

from carla.client import make_carla_client, VehicleControl
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
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
def run_carla_train(total_frames, model, device, optimizer, history, save_images, vehicles, pedestians) :
    with make_carla_client("localhost", 2000) as client:
        print('carla client connected')
        # setting up transform
        transform_list = []
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        # loss values to return
        reg_loss_dagger = 0
        cls_loss_dagger = 0
        # frames trained
        trained_frames = 0
        for episode in range(10000):
            # dagger end
            if trained_frames >= total_frames:
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
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            print("starting new episode ({})...".format(episode))
            client.start_episode(player_start)
            
            frames = torch.zeros(1, 3*history, 256, 256).float().to(device)
            # execute frames
            for frame_index in range(1000):
                measurements, sensor_data = client.read_data()
                
                # checking whether the episode should end (i.e. car crash or fucked up stuff)
                # measurements.player_measurements.collision_pedestrians doesn't matter. fuck pedestrians
                # print("{} type {}".format(measurements.player_measurements.collision_vehicles,type(measurements.player_measurements.collision_vehicles)))
                if  measurements.player_measurements.collision_vehicles > 0 \
                    or measurements.player_measurements.collision_other > 0 \
                    or measurements.player_measurements.intersection_otherlane > 0.30 \
                    or measurements.player_measurements.intersection_offroad > 0.30 \
                    or measurements.player_measurements.autopilot_control.hand_brake \
                    or measurements.player_measurements.autopilot_control.reverse :
                    break

                # getting expert controls
                control = measurements.player_measurements.autopilot_control
                expert = np.ndarray(shape=(3,), dtype = np.float32)
                control.steer = control.steer if np.abs(control.steer)>1e-3 else 0
                expert[0] = control.steer
                expert[1] = control.throttle
                expert[2] = control.brake
                # convering current frame
                frame = sensor_data['RGBFront'].data
                frame = np.transpose(frame, (1, 0, 2))
                frame = transform(frame).float().to(device)
                
                # if this is the first frame, fill the history buffer with the current frame
                # otherwise shift
                if frame_index ==0 :
                    for i in range(history) :
                        frames[0, i*3:(i+1)*3] = frame
                else :
                    frames[0, 3:] = frames[0, 0:-3]
                    frames[0, :3] = frame
                
                # getting agent predictions
                model.net.eval()
                pred_cls, pred_reg  = model.predict(frames)
                pred_cls = torch.argmax(pred_cls)
                pred_cls = pred_cls.item()
                pred_reg = pred_reg.item()
                agent = label_to_action_dobule(pred_cls)
                agent[0] = pred_reg

                # sending back agent's controls
                control = VehicleControl()
                control.steer = agent[0]
                control.throttle = agent[1] if measurements.player_measurements.forward_speed * 3.6 <=40 else 0
                control.brake = agent[2]
                control.hand_brake = False
                control.reverse = False
                client.send_control(control)

                # comparing controls (should we train this ?)
                if compare_controls(expert=expert, agent=agent) :
                    print_over_same_line("dagger frame {}/{}".format(trained_frames,total_frames))
                    label = 0 if expert[1] > 0 else \
                            1 if expert[2] > 0 else 2
                    label = torch.LongTensor([label])
                    steer = torch.Tensor([expert[0]])

                    model.net.train()
                    optimizer.zero_grad()
                    pred_cls, pred_reg  = model.net(frames)
                    loss_cls = classification_loss(pred_cls, label)
                    loss_reg = regression_loss(pred_reg, steer)
                    loss_cls.backward(retain_graph=True)
                    loss_reg.backward()
                    reg_loss_dagger += loss_reg.item()
                    cls_loss_dagger += loss_cls.item()
                    optimizer.step()
                    trained_frames +=1

            
        return reg_loss_dagger, cls_loss_dagger

def dagger(frames, model, device, optimizer, history, weather, vehicles, pedestians):
    while True:
        try:
            reg_loss_dagger, cls_loss_dagger = run_carla_train(total_frames=frames, model=model, device=device, optimizer=optimizer, history=history,
                                                        weather=weather, vehicles=vehicles, pedestians=pedestians)
            print('Done.')
            return reg_loss_dagger, cls_loss_dagger
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
