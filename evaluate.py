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
from utils import label_to_action

from server.PythonClient.carla.client import make_carla_client, VehicleControl
from server.PythonClient.carla.sensor import Camera, Lidar
from server.PythonClient.carla.settings import CarlaSettings
from server.PythonClient.carla.tcp import TCPConnectionError
from server.PythonClient.carla.util import print_over_same_line
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
def run_carla_eval(number_of_episodes, frames_per_episode, model, device, history, save_images, weather, vehicles, pedestians) :
    with make_carla_client("localhost", 2000) as client:
        print('carla client connected')
        # setting up transform
        transform_list = []
        # transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)

        # statistics to return
        avg_collision_vehicle = []
        avg_collision_pedestrian = []
        avg_collision_other = []
        avg_intersection_otherlane = []
        avg_intersection_offroad = []
        for episode in range(number_of_episodes):
            # settings
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
            camera.set_image_size(640, 480)
            camera.set(FOV=90.0)
            camera.set_position(1.65, 0, 1.30)
            camera.set_rotation(pitch=0, yaw=0, roll=0)
            settings.add_sensor(camera)
            
            
            # choosing starting position, starting episode
            scene = client.load_settings(settings)
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            print("starting new episode ({})...".format(episode))
            client.start_episode(player_start)
            
            frames = torch.zeros(1, 3*history, 640, 480).float().to(device)

            collision_vehicle = 0
            collision_pedestrian = 0
            collision_other = 0
            intersection_otherlane = 0
            intersection_offroad = 0
            # execute frames
            for frame_index in range(frames_per_episode):
                measurements, sensor_data = client.read_data()
                
                #print_measurements(measurements)
                for name, measurement in sensor_data.items():
                    # capture one episode
                    if save_images:
                        filename ="{}_e{:02d}_f{:03d}".format(name, episode, frame_index)
                        measurement.save_to_disk(os.path.join("./data",filename))

                
                # SDL_VIDEODRIVER = offscreen
                # getting expert controls
                control = measurements.player_measurements.autopilot_control
                expert = np.ndarray(shape=(3,), dtype = np.float32)
                control.steer = control.steer if np.abs(control.steer)>5e-4 else 0
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
                out = model.predict(frames)
                out = torch.argmax(out)
                out = out.item()
                agent = label_to_action(out)

                # sending back agent's controls
                control = VehicleControl()
                control.steer = agent[0]
                control.throttle = agent[1]
                control.brake = agent[2]
                control.hand_brake = False
                control.reverse = False
                client.send_control(control)

                # frame statistics
                collision_vehicle += measurements.player_measurements.collision_vehicles
                collision_pedestrian += measurements.player_measurements.collision_pedestrians
                collision_other += measurements.player_measurements.collision_other
                intersection_otherlane += 100 * measurements.player_measurements.intersection_otherlane
                intersection_offroad += 100 * measurements.player_measurements.intersection_offroad
            
            avg_collision_vehicle.append(collision_vehicle / frames_per_episode)
            avg_collision_pedestrian.append(collision_pedestrian / frames_per_episode)
            avg_collision_other.append(collision_other / frames_per_episode)
            avg_intersection_otherlane.append(intersection_otherlane / frames_per_episode)
            avg_intersection_offroad.append(intersection_offroad / frames_per_episode)
            
        return avg_collision_vehicle, avg_collision_pedestrian, avg_collision_other, avg_intersection_otherlane, avg_intersection_offroad


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)

def evaluate_model(episodes, frames, model, device, history, save_images, weather, vehicles, pedestians):
    while True:
        try:
            acv, acp, aco, aiol, aior = run_carla_eval(number_of_episodes=episodes, frames_per_episode=frames, model=model, device=device, history=history,
                                                       save_images=save_images, weather=weather, vehicles=vehicles, pedestians=pedestians)
            print('Done.')
            return acv, acp, aco, aiol, aior
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == "__main__":
    print("starting carla in server mode")
    my_env = os.environ.copy()
    my_env["SDL_VIDEODRIVER"] = "offscreen"
    FNULL = open(os.devnull, 'w')
    subprocess.Popen(['.././CarlaUE4.sh', '-benchmark', '-fps=20', '-carla-server', '-windowed', '-ResX=16', 'ResY=9'],stdout=FNULL, stderr=FNULL, env=my_env)
    print("done")

    device = torch.device('cpu')
    agent = CBCAgent(device=device, history=1)
    agent.net.load_state_dict(torch.load('snaps/july1_1_model_30'))
    acv, acp, aco, aiol, aior = evaluate_model(10,250,agent,device,1,True,1,20,20)
    for i in range(10):
        os.system("ffmpeg -r 20 -i data/RGBFront_e{:02d}_f%03d.png -b 500000  data/episode_{}.mp4".format(i,i))

    print("avg collision vehicle {}".format(sum(acv)/len(acv)))
    print("avg collision pedestrain {}".format(sum(acp)/len(acp)))
    print("avg collision other {}".format(sum(aco)/len(aco)))
    print("avg intersection otherlane {}".format(sum(aiol)/len(aiol)))
    print("avg intersection offroad {}".format(sum(aior)/len(aior)))
                                