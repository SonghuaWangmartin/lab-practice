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
from env.gymInvertedPendulum import InvertedPendulumEnv
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from collections import deque




gamma = 0.99
tau = 0.001
buffersize = 100000
noiseDacayStep = buffersize
batchsize =  64
minVar = 0
actorlearningRate = 0.0001 
criticlearningRate = 0.001
actornumberlayers = 256
criticnumberlayers = 256
paramUpdateFrequency = 1
noiseDecay = .9995
initnoisevar = 5

EPISODE = 3000
maxTimeStep = 1000
Env_name = 'InvertedPendulum-v1'
env = env_norm(gym.make(Env_name))
env = env.unwrapped

def main():

    statedim = env.observation_space.shape[0]
    actiondim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2
    replaybuffer = deque(maxlen=buffersize)
    
    totalrewards = []
    meanreward = []
    trajectory = []
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
    runstep = 0

    noisevar = initnoisevar
    for episode in range(EPISODE):
        state  = env.reset()
        done = False
        rewards = 0
        for t in range(maxTimeStep):
            noise,noisevar = getnoise(runstep,noisevar)
            noiseaction = getnoiseaction(state,noise)
            nextstate,reward,done,info = env.step(noiseaction)
            learn(replaybuffer,state, noiseaction, nextstate,reward)
            env.render()
            trajectory.append((state, noiseaction, nextstate,reward))
            rewards += reward
            state = nextstate
            runstep += 1
            if t == maxTimeStep-1:
                totalreward.append(rewards)
                print('episode: ',episode,'reward:',rewards, 'noisevar',noisevar,'runstep',runstep)
    if episode % 100 == 0:
        meanreward.append(np.mean(totalreward))
        print('episode: ',episode,'meanreward:',np.mean(totalreward))
        totalreward = []
        
    episode = 100*(np.arange(len(meanreward)))
    plt.plot(episode,meanreward)
    plt.xlabel('episode')
    

if __name__ == '__main__':
    main()
    
    

    
    
    
    
    
    
    
    
    