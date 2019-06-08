import gym
import MADRaS



env = gym.make('Madras-v0', throttle=True, vision=True, visualise=True, port=60934, pid_assist=False)


num_episodes = 10000
num_steps = 5000


for i in range(num_episodes):
    env.reset()
    for i in range(num_steps):
        # u[0] = steer, u[1] = accel, u[2] = break
        a = policy(obs)
        obs1, rew, done, info = env.step(a)


