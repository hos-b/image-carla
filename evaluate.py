"""evaluation script"""

import argparse
import logging
import random
import time
import os
import numpy as np
import h5py
import subprocess
import csv

import torchvision.transforms as transforms
import torch
from utils import label_to_action_dobule

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
def run_carla_eval(number_of_episodes, frames_per_episode, model, device, history, save_images, weather, vehicles, pedestians, carla_port, ground_truth) :
    with make_carla_client("localhost", carla_port) as client:
        print('carla client connected')
        # setting up transform
        transform_list = []
        transform_list.append(transforms.ToPILImage())
        # transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)

        # statistics to return
        avg_collision_vehicle = []
        avg_collision_pedestrian = []
        avg_collision_other = []
        avg_intersection_otherlane = []
        avg_intersection_offroad = []
        avg_distance_traveled = []
        avg_accidents_vehicle = []
        avg_accidents_pedestrain = []
        avg_accidents_other = []
        for episode in range(number_of_episodes):
            # settings
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=vehicles,
                NumberOfPedestrians=pedestians,
                WeatherId=weather, 
                QualityLevel='Epic') # QualityLevel=args.quality_level
            settings.randomize_seeds()

            camera = Camera('RGBFront', PostProcessing='SceneFinal')
            camera.set_image_size(512, 512)
            camera.set(FOV=120.0)
            # camera.set_position(1.65, 0, 1.30) < OLD
            camera.set_position(2.0, 0, 1.60)
            camera.set_rotation(roll=0, pitch=-10, yaw=0)
            settings.add_sensor(camera)
            
            
            # choosing starting position, starting episode
            scene = client.load_settings(settings)
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            print("running eval episode {}/{} in location {}".format(episode+1,number_of_episodes,player_start))
            client.start_episode(player_start)
            
            frames = torch.zeros(1, 3*history, 256, 256).float().to(device)

            # variables for normal metrics
            collision_vehicle = 0
            collision_pedestrian = 0
            collision_other = 0
            intersection_otherlane = 0
            intersection_offroad = 0
            # variables for distance traveled metric
            distance_traveled = 0
            last_position = 0
            # variables for accident count metrics
            last_accident_vehicle = 0
            last_accident_pedestrain = 0
            last_accident_other = 0
            accident_count_vehicle = 0
            accident_count_pedestrain = 0
            accident_count_other = 0
            # execute frames
            for frame_index in range(frames_per_episode):
                print_over_same_line("frame {}/{}".format(frame_index+1,frames_per_episode))
                measurements, sensor_data = client.read_data()
                
                #print_measurements(measurements)
                for name, measurement in sensor_data.items():
                    # capture one episode
                    if save_images:
                        filename ="{}_e{:02d}_f{:03d}".format(name, episode, frame_index)
                        measurement.save_to_disk(os.path.join("./data",filename))

                # convering current frame
                frame = sensor_data['RGBFront'].data
                frame = np.transpose(frame, (1, 0, 2))
                frame = transform(frame).float().to(device)
                
                # if first frame, fill the history  && set the last_pose to inital one
                # otherwise shift frames            && increase distance_traveled        && check las
                if frame_index ==0 :
                    last_position = measurements.player_measurements.transform.location
                    for i in range(history) :
                        frames[0, i*3:(i+1)*3] = frame
                else :
                    frames[0, 3:] = frames[0, 0:-3]
                    frames[0, :3] = frame
                    # updating variables for metrics
                    distance_traveled += distance_3d(last_position, measurements.player_measurements.transform.location)
                    last_position = measurements.player_measurements.transform.location
                    accident_count_vehicle += 1 if (last_accident_vehicle==0 and measurements.player_measurements.collision_vehicles>0) else 0
                    accident_count_pedestrain +=1 if (last_accident_pedestrain==0 and measurements.player_measurements.collision_pedestrians>0) else 0
                    accident_count_other+=1 if (last_accident_other==0 and measurements.player_measurements.collision_other>0) else 0
                    last_accident_vehicle = measurements.player_measurements.collision_vehicles
                    last_accident_pedestrain = measurements.player_measurements.collision_pedestrians
                    last_accident_other = measurements.player_measurements.collision_other
                
                control = VehicleControl()
                # use autopilot
                if ground_truth : 
                    control = measurements.player_measurements.autopilot_control
                else :
                    # getting agent predictions
                    model.net.eval()
                    pred_cls, pred_reg  = model.predict(frames)
                    pred_cls = torch.argmax(pred_cls)
                    pred_cls = pred_cls.item()
                    pred_reg = pred_reg.item()
                    agent = label_to_action_dobule(pred_cls, pred_reg)
                
                    # sending back agent's controls
                    control.steer = pred_reg
                    control.throttle = agent[1] if measurements.player_measurements.forward_speed * 3.6 <=30 else 0
                    control.brake = agent[2]
                    control.hand_brake = False
                    control.reverse = False

                client.send_control(control)

                # frame statistics
                collision_vehicle += last_accident_vehicle
                collision_pedestrian += last_accident_pedestrain
                collision_other += last_accident_other
                intersection_otherlane += 100 * measurements.player_measurements.intersection_otherlane
                intersection_offroad += 100 * measurements.player_measurements.intersection_offroad
            
            avg_collision_vehicle.append(collision_vehicle / frames_per_episode)
            avg_collision_pedestrian.append(collision_pedestrian / frames_per_episode)
            avg_collision_other.append(collision_other / frames_per_episode)
            avg_intersection_otherlane.append(intersection_otherlane / frames_per_episode)
            avg_intersection_offroad.append(intersection_offroad / frames_per_episode)
            avg_distance_traveled.append(distance_traveled)
            avg_accidents_vehicle.append(accident_count_vehicle)
            avg_accidents_pedestrain.append(accident_count_pedestrain)
            avg_accidents_other.append(accident_count_other)
            
        return avg_collision_vehicle, avg_collision_pedestrian, avg_collision_other, avg_intersection_otherlane, avg_intersection_offroad, avg_distance_traveled, avg_accidents_vehicle, avg_accidents_pedestrain, avg_accidents_other

def evaluate_model(episodes, frames, model, device, history, save_images, weather, vehicles, pedestians,carla_port, ground_truth):
    while True:
        try:
            acv, acp, aco, aiol, aior, adt, aav, aap, aao = run_carla_eval(number_of_episodes=episodes, frames_per_episode=frames, model=model, device=device, history=history,
                                                       save_images=save_images, weather=weather, vehicles=vehicles, pedestians=pedestians,carla_port=carla_port, ground_truth=ground_truth)
            print('done')
            return acv, acp, aco, aiol, aior, adt, aav, aap, aao
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == "__main__":
    random.seed(30)
    print("starting carla in server mode")
    my_env = os.environ.copy()
    my_env["DISPLAY"] = ""
    FNULL = open(os.devnull, 'w')
    subprocess.Popen(['server/./CarlaUE4.sh', '-benchmark', '-fps=20', '-carla-server', '-windowed', '/Game/Maps/Town02', 
                      '-opengl', '-ResX=16', '-ResY=9','-world-port=5000'], stdout=FNULL, stderr=FNULL, env=my_env)
    print("done")

    device = torch.device('cuda')
    agent = CBCAgent(device=device, history=3, name='efficient-double-large')
    
    for model_name in ['dnet_h3w_16th_HD_model_32', 'dnet_h3w_16th_HD_nodag_model_32'] :
        print("evaluating {}".format(model_name))
        agent.net.load_state_dict(torch.load("snaps/{}".format(model_name)))
        agent.net.to(device)
        _, _, _, aiol, aior, adt, aav, aap, aao = evaluate_model(3,500,agent,device,3,False,1,30,0,carla_port=5000, ground_truth=False)
        # carl.close()
        # os.system("mkdir data/{}".format(model_name))
        # os.system("cp snaps/{} data/{}".format(model_name, model_name))
    
        # for i in range(10):
        #    os.system("ffmpeg -r 20 -i data/RGBFront_e{:02d}_f%03d.png -b 500000  data/{}/episode_{}.mp4".format(i,model_name,i))
        # os.system("rm -f data/*.png")
        # os.system("nautilus ./data")
        
        with open('csv/aiol_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(aiol)
        with open('csv/aior_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(aior)
        with open('csv/adt_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(adt)
        with open('csv/aav_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(aav)
        with open('csv/aap_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(aap)
        with open('csv/aao_{}.csv'.format(model_name), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(aao)