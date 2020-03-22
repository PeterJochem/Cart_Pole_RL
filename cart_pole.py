import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import copy

env = gym.make('CartPole-v0')

# Hyper parameters
learningRate = 0.80
discountRate = 0.99
exploreRate = 1.0
exploreDecayRate = 0.95

replaySize = 500
replayBuffer = []
replayBufferLabels = []

Q_value = Sequential()
Q_value.add(Dense(24, input_dim = 4, activation='relu'))
Q_value.add(Dense(24, activation='relu'))
Q_value.add(Dense(2, activation='linear'))
Q_value.compile(loss='mse',  optimizer = Adam(lr=0.001)   )

observation = env.reset()
observationPrior = observation
action = 0

totalReward = 0
numEntries = 0.0

episodeNum = 0

for i in range(1000000):
    env.render()

    if ( exploreRate > random.random() ):
        action = env.action_space.sample()
    else:
        # returns the index of the maximum element
        action = np.argmax( Q_value.predict( np.array([observation] ) )[0] ) 

    observation, reward, done, info = env.step(action) 
    
    if ( done == True ):
        reward = -reward

    totalReward = totalReward + reward
    numEntries += 1
    
    value = Q_value.predict( np.array([observationPrior]) )[0]
    # Add the experience to the list of data
    replayBuffer.append(observationPrior)
    
    value[action] = reward

    if ( (action == 0) and (done == False) ):
            #value[0] = value[0] + (learningRate * (reward + discountRate * np.amax(Q_value.predict(np.array([observation])) ) - value[0] ) )
            value[0] = reward + discountRate * np.amax(Q_value.predict(np.array([observation]) )[0] )
    elif ( done == False):
            #value[1] = value[1] + (learningRate * (reward + discountRate * np.amax(Q_value.predict(np.array([observation])) ) - value[1] ) )
            value[1] = reward + discountRate * np.amax(Q_value.predict(np.array([observation]) )[0] )


    replayBufferLabels.append(value)

    observationPrior = copy.deepcopy(observation)

    if (done == True):
        observation = env.reset()
        observationPrior = copy.deepcopy(observation)
        episodeNum = episodeNum + 1

    if ( i % replaySize == 0):
        # print(totalReward/numEntries)
        totalReward = 0
        numEntries = 0.0

        np.random.shuffle(replayBuffer)
        np.random.shuffle(replayBufferLabels)

        # Update the Q-Table
        Q_value.fit( np.array(replayBuffer), np.array(replayBufferLabels), verbose = 0)
        
        #replayBuffer = []
        #replayBufferLabels = []
        
        exploreRate = exploreRate * exploreDecayRate




env.close()
