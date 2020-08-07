"""
Created on Sat Jun 13 14:14:08 2020

@author: martin
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.simpleDQL import ReplaceParameters,epsilonDec,BuildModel,Learn,TrainModel,TrainDQNmodel
from env.MountainCarDiscrete import MtCarDiscreteEnvSetup,visualizeMtCarDiscrete,resetMtCarDiscrete,MtCarDiscreteTransition,MtCarDiscreteReward,MtCarDiscreteIsTerminal
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from collections import deque

gamma = 0.99
buffersize = 50000
batchsize =  128
initepsilon = 1
minepsilon = 0.01
epsilondec = 0.0001
learningRate = 0.001 
numberlayers = 256
replaceiter = 1000

ENV_NAME = 'MountainCar-v0'
EPISODE = 4501

def main():
    env = MtCarDiscreteEnvSetup()        
    visualize = visualizeMtCarDiscrete() 
    reset = resetMtCarDiscrete(1234)         
    transition = MtCarDiscreteTransition()  
    rewardMtCar = MtCarDiscreteReward()        
    isterminal = MtCarDiscreteIsTerminal()
    
    statesdim = env.observation_space.shape[0]
    actiondim = env.action_space.n
    replaybuffer = deque(maxlen=buffersize)
    runepsilon = initepsilon
    totalrewards = []
    meanreward = []
    trajectory = []
    totalreward = []
    
    buildmodel = BuildModel(statesdim,actiondim)
    Writer,DQNmodel = buildmodel(numberlayers)
    replaceParameters = ReplaceParameters(replaceiter)
    trainModel = TrainModel(learningRate, gamma,Writer)
    trainDQNmodel = TrainDQNmodel(replaceParameters, trainModel, DQNmodel)
    learn = Learn(buffersize,batchsize,trainDQNmodel,actiondim)
    
    
    for episode in range(EPISODE):
        state  = reset()
        rewards = 0
        runtime = 0
        while True:
            action = learn.Getaction(DQNmodel,runepsilon,state)  
            nextstate=transition(state, action)
            done = isterminal(nextstate)
            reward = rewardMtCar(state,action,nextstate,done)
            learn.ReplayMemory(replaybuffer,state, action, reward, nextstate,done)
            trajectory.append((state, action, reward, nextstate))
            rewards += reward
            state = nextstate
            runtime += 1
            if runtime == 200:
                totalrewards.append(rewards)
                totalreward.append(rewards)
                runtime = 0
                print('episode: ',episode,'reward:',rewards,'epsilon:',runepsilon)
                break
            if done:
                totalrewards.append(rewards)
                totalreward.append(rewards)
                print('episode: ',episode,'reward:',rewards,'epsilon:',runepsilon)
                break
        runepsilon = epsilonDec(runepsilon,minepsilon,epsilondec)
        if episode % 100 == 0:
            meanreward.append(np.mean(totalreward))
            print('episode: ',episode,'meanreward:',np.mean(totalreward))
            totalreward = []
    episode = 100*(np.arange(len(meanreward)))
    plt.plot(episode,meanreward)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.ylim([-200,-50])
    plt.show()

# save Model
    modelIndex = 0
    DQNFixedParam = {'DQNmodel':modelIndex}
    parameters = {'env': ENV_NAME, 'Eps': EPISODE,  'batch': batchsize,'buffersize': buffersize,
                  'gam': gamma, 'learningRate': learningRate,
                  'replaceiter': replaceiter, 'epsilondec': epsilondec, 'minepsilon': minepsilon, 'initepsilon': initepsilon}

    modelSaveDirectory = "/path/to/logs/trainedDQNModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, DQNFixedParam)
    savePathDQN = getSavePath(parameters)

    with DQNmodel.as_default():
        saveVariables(DQNmodel, savePathDQN)
        
    dirName = os.path.dirname(__file__)
    trajectoryPath = os.path.join(dirName,'trajectory', 'mountCarTrajectory.pickle')
    saveToPickle(trajectory, trajectoryPath)
    
if __name__ == '__main__':
  main()
