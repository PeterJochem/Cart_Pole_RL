import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import copy
from trainingInstance import trainingInstance

# Creates enviroment from OpenAI gym
env = gym.make('CartPole-v0')

# Hyper parameters
learningRate = 0.80
discountRate = 0.99
exploreRate = 1.0
exploreDecayRate = 0.95

# Create the neural network
Q_value = Sequential()
Q_value.add(Dense(48, input_dim = 4, activation='relu'))
Q_value.add(Dense(24, activation='relu'))
Q_value.add(Dense(2, activation='linear'))
Q_value.compile(loss='mse',  optimizer = Adam(lr = 0.001) )

observation = env.reset()
observation = np.reshape(observation, [1, 4])
observationPrior = copy.deepcopy(observation)
action = 0

totalReward = 0
gameNum = 0

# This is the memory replay buffer
memory = []

while (gameNum < 50000):
    
    env.render()

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
    
    if ( done == True ):
        reward = -100

    value = Q_value.predict(observationPrior)
    
    value[0][action] = reward

    if ( (action == 0) and (done == False) ):
            value[0][0] = reward + discountRate * np.amax(Q_value.predict(observation)[0] )
    elif ( done == False ):
            value[0][1] = reward + discountRate * np.amax(Q_value.predict(observation)[0] ) 
    
    memory.append( trainingInstance(observationPrior, observation, reward, action, done, gameNum)  )
    
    batchSize = 20 #10

    batch = memory
    if ( len(memory) < batchSize):
        batch = memory
    else:
        batch = random.sample(memory, batchSize)

    for i in range(len(batch) ):
        value = Q_value.predict(batch[i].observationPrior)

        value[0][action] = batch[i].reward

        if ( (batch[i].action == 0) and (batch[i].done == False) ):
            value[0][0] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )
        elif ( memory[i].done == False ):
            value[0][1] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )

        Q_value.fit( batch[i].observationPrior, value, epochs = 1, verbose = 0)
    
    exploreRate = exploreRate * exploreDecayRate
    exploreRate = max(exploreRate, 0.01)

    observationPrior = copy.deepcopy(observation)

    if (done == True):
        observation = env.reset()
        observation = np.reshape(observation, [1, 4])
        observationPrior = copy.deepcopy(observation)
        
        gameNum = gameNum + 1

        print("The total reward for game " + str(gameNum) +  " is " + str(totalReward) )
        totalReward = 0
    
    
    if ( (gameNum % 1000) == 0 ):
        batchSize = 20

        batch = memory
        if ( len(memory) < batchSize):
            batch = memory
        else:
            batch = random.sample(memory, batchSize)

        for i in range(len(batch) ):
            value = Q_value.predict(batch[i].observationPrior)

            value[0][action] = batch[i].reward

            if ( (batch[i].action == 0) and (batch[i].done == False) ):
                    value[0][0] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )
            elif ( memory[i].done == False ):
                    value[0][1] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )

            Q_value.fit( batch[i].observationPrior, value, epochs = 1, verbose = 0)        
            

        exploreRate = exploreRate * exploreDecayRate
        exploreRate = max(exploreRate, 0.01)
        
    if ( gameNum % 10 == 0):    
        memory = []
        exploreRate = 0.3

# This will display the learned agent to the user
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
