import numpy as np
import gym
import MADRaS
import sys
import time
from pyglet.window import key

def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  action[0] = -1.0
    if k == key.RIGHT: action[0] = +1.0
    if k == key.UP:    action[1] = +1.0
    if k == key.DOWN:  action[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and action[0] == -1.0: action[0] = 0.0
    if k == key.RIGHT and action[0] == +1.0: action[0] = 0.0
    if k == key.UP:    action[1] = 0.0
    if k == key.DOWN:  action[2] = 0.0


if __name__ == "__main__":
    
    # set up the environment
    env = gym.make('Madras-v0', throttle=True, vision=True, visualise=True, port=3001, pid_assist=False)
    print ("env port : {}".format(env.port))
    #env.viewer.window.on_key_press = key_press
    #env.viewer.window.on_key_release = key_release
    action = np.array([0.0, 0.0, 0.0]).astype('float32')
    episode = 0
    
    while True:
        obs = env.reset()
        step = 0
        episode_reward = 0
        while True:
            # u[0] = steer, u[1] = accel, u[2] = brake
            obs, rew, done, info = env.step(action)
            
            episode_reward += rew
            step +=1

            env.render()

            if done :
                break

        episode += 1

    env.close()