"""Data collection script"""

import argparse
import logging
import random
import time
import os
import numpy as np
import h5py

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

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
def run_carla_client(args, number_of_episodes=10, frames_per_episode=500, starting_episode=0):
    with make_carla_client("localhost", 2000) as client:
        print('carla client connected')
        # setting up data type
        imitation_type = np.dtype([('image', np.uint8, (640, 480, 3)), ('label', np.float32, 5)])
        print("datatype :\n{}".format(imitation_type))
        # total frames collected this run
        total_frames = 0

        for episode in range(starting_episode, number_of_episodes):
            # flag to skip the initial frames where the car doesn't move
            record = False
            # settings
            settings = CarlaSettings()
            if args.settings_filepath is None:
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=20,
                    NumberOfPedestrians=40,
                    WeatherId=1, # clear noon
                    QualityLevel='Low') # QualityLevel=args.quality_level
                settings.randomize_seeds()

                camera = Camera('RGBFront', PostProcessing='SceneFinal')
                camera.set_image_size(640, 480)
                camera.set(FOV=90.0)
                camera.set_position(1.65, 0, 1.30)
                camera.set_rotation(pitch=0, yaw=0, roll=0)
                settings.add_sensor(camera)
            else:
                # load settings from file
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()
            
            # choosing starting position, starting episode
            scene = client.load_settings(settings)
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            print("starting new episode ({})... {} frames saved".format(episode, total_frames))
            client.start_episode(player_start)

            # keeping track of the frames that are actually being saved
            frame_index = 0
            # new episode, new dataset
            dataset = hdf5_file.create_dataset("episode_{}".format(episode),shape =(1,), maxshape=(None,), chunks=(1,), compression="lzf", dtype=imitation_type)
            # execute frames
            for frame in range(0, frames_per_episode):
                measurements, sensor_data = client.read_data()
                if frame==0 :
                    initial_pose = measurements.player_measurements.transform.location
                elif not record: 
                    distance = distance_3d(initial_pose, measurements.player_measurements.transform.location)
                    record = distance > 1.5
                else:
                    frame_index+=1
                
                # print_measurements(measurements)
                for name, measurement in sensor_data.items():
                    if record and args.save_images_to_disk:
                        filename ="{}_e{}_f{}".format(name, episode, frame_index)
                        measurement.save_to_disk(os.path.join("./data",filename))

                # getting autopilot controls
                control = measurements.player_measurements.autopilot_control
                # SDL_VIDEODRIVER = offscreen
                if record:
                    label = np.ndarray(shape=(5,), dtype = float)
                    control.steer = control.steer if np.abs(control.steer)>5e-4 else 0
                    label[0] = control.steer
                    label[1] = control.throttle
                    label[2] = control.brake
                    label[3] = 1 if control.hand_brake else 0
                    label[4] = 1 if control.reverse else 0

                    frame = sensor_data['RGBFront'].data
                    frame = np.transpose(frame, (1, 0, 2))
                    data = np.array([(frame, label)], dtype=imitation_type)
                    dataset.resize(frame_index+2, axis=0)
                    dataset[frame_index] = data
                    total_frames+=1

                client.send_control(control)


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


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()
    print ("settings_filepath : {}\nsave to disk : {}".format(args.settings_filepath, args.save_images_to_disk))
    print ("quality : {}".format(args.quality_level))

    dslist = list(hdf5_file.keys())
    start = 0
    if len(dslist) != 0:
        print("hdf5 file not empty, resuming data collection")
        ch = dslist[-1]
        ch = ch[-1]
        start = int(ch) + 1
        print("last episode : {}".format(start))

    while True:
        try:
            run_carla_client(args, number_of_episodes=210, frames_per_episode=700, starting_episode=start)
            print('Done.')
            return
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        hdf5_file = h5py.File(os.path.join("/home/hosein/part","carla_dataset.hdf5"), "a")
        main()
    except KeyboardInterrupt:
        print("closing hdf5 file")
        hdf5_file.close()
        print('\nCancelled by user. Bye!')
