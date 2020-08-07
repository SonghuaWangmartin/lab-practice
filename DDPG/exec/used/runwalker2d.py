

import os
import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.DDPG import *
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from collections import deque



gamma = 0.99
tau = 0.001
buffersize = 500000
initbuffer = 30000
noiseDacayStep = 10000
batchsize =  256
minVar = 0.001
actorlearningRate = 0.0001 
criticlearningRate = 0.001
actornumberlayers = 256
criticnumberlayers = 256
noiseDecay = 0.001
initnoisevar = 2
EPISODE = 2000
maxTimeStep = 1000
Env_name = 'Walker2d-v1'
env = gym.make(Env_name)
env = env.unwrapped
env.seed(1)

def main():
    statedim = env.observation_space.shape[0]
    actiondim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2
    replaybuffer = deque(maxlen=buffersize)
    
    meanreward = []
    trajectory = []
    totalreward = []
    totalrewards = []
    
    buildActorModel = BuildActorModel(statedim, actiondim, actionBound)
    actorWriter, actorModel = buildActorModel(actornumberlayers)
    
    buildCriticModel = BuildCriticModel(statedim, actiondim)
    criticWriter, criticModel = buildCriticModel(criticnumberlayers)

    trainCritic = TrainCritic(criticlearningRate, gamma, criticWriter)
    trainActor = TrainActor(actorlearningRate, actorWriter)
    updateParameters = UpdateParameters(tau)
    trainddpgModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel, criticModel)
    
    getnoise = GetNoise(noiseDecay,minVar,noiseDacayStep,initnoisevar)
    getnoiseaction = GetNoiseAction(actorModel,actionLow, actionHigh)
    learn = Learn(buffersize,batchsize,trainddpgModels)
    state = env.reset()
    bufferfill = 0
    replaybuffer = fillbuffer(initbuffer,bufferfill,env,replaybuffer,state)
            
    actorModel= ReplaceParameters(actorModel)
    criticModel= ReplaceParameters(criticModel)
    runstep = 0
    for episode in range(EPISODE):
        state  = env.reset()
        done = False
        rewards = 0
        for j in range(maxTimeStep):
            env.render()
            noise = getnoise(runstep)
            noiseaction = getnoiseaction(state,noise)
            nextstate,reward,done,info = env.step(noiseaction)
            learn(replaybuffer,state, noiseaction, nextstate,reward)
            trajectory.append((state, noiseaction, nextstate,reward))
            rewards += reward
            state = nextstate
            runstep += 1
            if j == maxTimeStep-1:
                totalrewards.append(rewards)
                totalreward.append(rewards)
                print('episode: ',episode,'reward:',rewards,'runstep',runstep)
        if episode % 100 == 0:
            meanreward.append(np.mean(totalreward))
            print('episode: ',episode,'meanreward:',np.mean(totalreward))
            totalreward = []
    episodes = 100*(np.arange(len(meanreward)))
    plt.plot(episodes,meanreward)
    plt.xlabel('episode')
    plt.show()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    