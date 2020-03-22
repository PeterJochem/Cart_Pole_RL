import gym
import numpy as np

env = gym.make('CartPole-v0')

env.reset()

observation = env.reset()

for i in range(1000):
    env.render()

    # print("")
    # print(env.action_space.sample() )
    # env.step(env.action_space.sample()) # take a random action
    if ( (observation[3] < 0) ):
        action = 1
    else:
        action = 0
 
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action) 
    
    # print(observation)


env.close()
