# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:14:08 2020

@author: martin
"""


import os
import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.DDPG import *
from collections import deque
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle


    

gamma = 0.95
tau = 0.01
buffersize = 10000
noiseDacayStep = 10000
batchsize =  64
minVar = 0
actorlearningRate = 0.001 
criticlearningRate = 0.001
actornumberlayers = 256
criticnumberlayers = 256
initnoisevar = 3
noiseDecay = .9995

EPISODE = 201
maxTimeStep = 200
Env_name = 'Pendulum-v0'
env = gym.make(Env_name)
env = env.unwrapped

def main():
    statedim = env.observation_space.shape[0]
    actiondim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2
    
    replaybuffer = deque(maxlen=buffersize)
    paramUpdateFrequency = 1
    totalrewards = []
    meanreward = []
    totalreward = []
    
    buildActorModel = BuildActorModel(statedim, actiondim, actionBound)
    actorWriter, actorModel = buildActorModel(actornumberlayers)
    
    buildCriticModel = BuildCriticModel(statedim, actiondim)
    criticWriter, criticModel = buildCriticModel(criticnumberlayers)

    trainCritic = TrainCritic(criticlearningRate, gamma, criticWriter)
    trainActor = TrainActor(actorlearningRate, actorWriter)
    updateParameters = UpdateParameters(tau,paramUpdateFrequency)

    actorModel= ReplaceParameters(actorModel)
    criticModel= ReplaceParameters(criticModel)
    
    trainddpgModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel, criticModel)
    
    getnoise = GetNoise(noiseDecay,minVar,noiseDacayStep)
    getnoiseaction = GetNoiseAction(actorModel,actionLow, actionHigh)
    learn = Learn(buffersize,batchsize,trainddpgModels,actiondim)
    runtime = 0
    trajectory = []
    noisevar = initnoisevar
    for episode in range(EPISODE):
        state  = env.reset()
        rewards = 0
        for i in range(maxTimeStep):
            env.render()
            noise,noisevar = getnoise(runtime,noisevar)
            noiseaction = getnoiseaction(state,noise)
            nextstate,reward,done,info = env.step(noiseaction)
            learn(replaybuffer,state, noiseaction, nextstate,reward)
            trajectory.append((state, noiseaction, nextstate,reward))
            rewards += reward
            state = nextstate
            runtime += 1
            print(actionHigh,actionLow)
        if i == maxTimeStep-1:
            totalrewards.append(rewards)
            totalreward.append(rewards)
            print('episode: ',episode,'reward:',rewards, 'noisevar',noisevar)
            
        if episode % 100 == 0:
            meanreward.append(np.mean(totalreward))
            print('episode: ',episode,'meanreward:',np.mean(totalreward))
            totalreward = []
    plt.plot(range(EPISODE),totalrewards)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()
# save Model
    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'env': Env_name, 'Eps': EPISODE,  'batchsize': batchsize,'buffersize': buffersize,'maxTimeStep':maxTimeStep,
                  'gamma': gamma, 'actorlearningRate': actorlearningRate, 'criticlearningRate': criticlearningRate,
                  'tau': tau, 'noiseDecay': noiseDecay, 'minVar': minVar, 'initnoisevar': initnoisevar}

    modelSaveDirectory = "/path/to/logs/trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, actorFixedParam)
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, criticFixedParam)
    savePathDQN = getSavePath(parameters)

    with actorModel.as_default():
        saveVariables(actorModel, savePathDQN)
    with criticModel.as_default():
        saveVariables(criticModel, savePathDQN)
        
    dirName = os.path.dirname(__file__)
    trajectoryPath = os.path.join(dirName,'trajectory', 'HopperTrajectory.pickle')
    saveToPickle(trajectory, trajectoryPath)
    


 
    
if __name__ == '__main__':
  main()
