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
discountRate = 0.95
exploreRate = 1.0
exploreDecayRate = 0.9995

replaySize = 500
replayBuffer = []
replayBufferLabels = []

Q_value = Sequential()
Q_value.add(Dense(24, input_dim = 4, activation='relu'))
Q_value.add(Dense(24, activation='relu'))
Q_value.add(Dense(2, activation='linear'))
Q_value.compile(loss='mse',  optimizer = Adam(lr = 0.005) )

observation = env.reset()
observation = np.reshape(observation, [1, 4])
observationPrior = copy.deepcopy(observation)
action = 0

totalReward = 0
numEntries = 0.0

episodeNum = 0

gameNum = 0

while (gameNum < 50000):
    
    if ( exploreRate > random.random() ):
        action = env.action_space.sample()
    else:
        # returns the index of the maximum element
        action = np.argmax( Q_value.predict( observation)[0] ) 

    observation, reward, done, info = env.step(action) 
    observation = np.reshape(observation, [1, 4])

    if ( done == True ):
        reward = -1

    totalReward = totalReward + reward
    numEntries += 1
    
    if ( done == True ):
        reward = -100


    value = Q_value.predict(observationPrior)
    # value = np.reshape(value, [1, 2])
    # Add the experience to the list of data
    replayBuffer.append(observationPrior)
    
    value[0][action] = reward

    if ( (action == 0) and (done == False) ):
            #value[0] = value[0] + (learningRate * (reward + discountRate * np.amax(Q_value.predict(np.array([observation])) ) - value[0] ) )
            value[0][0] = reward + discountRate * np.amax(Q_value.predict(observation)[0] )
    elif ( done == False ):
            #value[1] = value[1] + (learningRate * (reward + discountRate * np.amax(Q_value.predict(np.array([observation])) ) - value[1] ) )
            value[0][1] = reward + discountRate * np.amax(Q_value.predict(observation)[0] ) 

    Q_value.fit( observationPrior, value, epochs = 1, verbose = 0)
    
    replayBufferLabels.append(value[0])

    observationPrior = copy.deepcopy(observation)

    if (done == True):
        observation = env.reset()
        observation = np.reshape(observation, [1, 4])
        observationPrior = copy.deepcopy(observation)
        episodeNum = episodeNum + 1
        gameNum = gameNum + 1

        print("The total reward for game " + str(gameNum) +  " is " + str(totalReward) )
        totalReward = 0
        numEntries = 0.0
    else:
        np.random.shuffle(replayBuffer)
        np.random.shuffle(replayBufferLabels)


    if ( (gameNum % 10) == 0 ):
        
        # Q_value.fit( replayBuffer, replayBufferLabels, epochs = 1, verbose = 0)
        # print( replayBufferLabels )

        replayBuffer = []
        replayBufferLabels = []

        exploreRate = exploreRate * exploreDecayRate
        exploreRate = max(exploreRate, 0.01)
    

observation = env.reset()
observation = np.reshape(observation, [1, 4])
while (True):
    env.render()
    observation = np.reshape(observation, [1, 4])
    action = np.argmax( Q_value.predict( observation)[0] )

    observation, reward, done, info = env.step(action)
    
    if ( done == True):
        observation = env.reset()
        observation = np.reshape(observation, [1, 4])


env.close()
