import os
import pandas as pd
import pylab as plt
import sys
import numpy as np
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from src.simpleDQL import ReplaceParameters,epsilonDec,BuildModel,Learn,TrainModel,TrainDQNmodel
from collections import deque, OrderedDict
from functionTools.loadSaveModel import GetSavePath, saveVariables
from env.Cartpole import CartPoleEnvSetup,visualizeCartpole,resetCartpole,CartPoletransition,CartPoleReward,isTerminal
os.environ['KMP_DUPLICATE_LIB_OK']='True'



ENV_NAME = 'Carpole-v0'
env = CartPoleEnvSetup()


class EvaluateLearningrateAndbatchsize:
    def __init__(self, fixedParameters, getSavePath, saveModel = True):
        self.fixedParameters = fixedParameters
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        learningRate = df.index.get_level_values('learningRate')[0]
        buffersize = df.index.get_level_values('buffersize')[0]
        
            
        visualize = visualizeCartpole() 
        reset = resetCartpole()         
        transition = CartPoletransition()  
        rewardcart = CartPoleReward()        
        isterminal = isTerminal()
        replaybuffer = deque(maxlen=int(buffersize))
        trajectory = []
        totalrewards = []
        averagerewards = []
    
        buildmodel = BuildModel(self.fixedParameters['stateDim'],self.fixedParameters['actionDim'])
        Writer,DQNmodel = buildmodel(self.fixedParameters['numberlayers'])
        replaceParameters = ReplaceParameters(self.fixedParameters['replaceiter'])
        trainModel = TrainModel(learningRate, self.fixedParameters['gamma'],Writer)
        trainDQNmodel = TrainDQNmodel(replaceParameters, trainModel, DQNmodel)
        learn = Learn(buffersize,self.fixedParameters['batchsize'],trainDQNmodel,self.fixedParameters['actionDim'])
        runepsilon = self.fixedParameters['initepsilon']
        
        
        
        for episode in range(self.fixedParameters['maxEpisode']):
            state  = reset()
            rewards = 0
            while True:
                visualize(state)
                runepsilon = epsilonDec(runepsilon,self.fixedParameters['minepsilon'],self.fixedParameters['epsilondec'])
                action = learn.Getaction(DQNmodel,runepsilon,state)  
                nextstate=transition(state, action)
                done = isterminal(nextstate)
                reward = rewardcart(state,action,nextstate,done)
                trajectory.append((state, action, reward, nextstate))
                learn.ReplayMemory(replaybuffer,state, action, reward, nextstate,done)
                rewards += reward
                state = nextstate
                if done:
                    totalrewards.append(rewards)
                    print('episode: ',episode,'reward:',rewards,'epsilon:',runepsilon)
                    break
            averagerewards.append(np.mean(totalrewards))
            print('episode:',episode,'meanreward:',np.mean(totalrewards))



        timeStep = list(range(len(averagerewards)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, averagerewards)})

        if self.saveModel:
            Parameters = {'learningRate': learningRate, 'buffersize': buffersize }
            modelPath = self.getSavePath(Parameters)
            with DQNmodel.as_default():
                saveVariables(DQNmodel, modelPath)

        return resultSe



def main():
        
    statesdim = env.observation_space.shape[0]
    actiondim = env.action_space.n
    
    fixedParameters = OrderedDict()
    fixedParameters['batchsize'] = 128
    fixedParameters['maxEpisode'] = 400
    fixedParameters['gamma'] = 0.9
    fixedParameters['numberlayers'] = 20
    fixedParameters['stateDim'] = statesdim
    fixedParameters['actionDim'] = actiondim
    fixedParameters['replaceiter'] = 100
    fixedParameters['initepsilon'] = 0.9
    fixedParameters['minepsilon'] = 0.1
    fixedParameters['epsilondec'] = 0.001


    learningRate = [0.1,0.01,0.001]
    buffersize = [1000,5000,20000]

    modelSaveDirectory = "/path/to/logs/trainedDQNModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    evaluateModel = EvaluateLearningrateAndbatchsize(fixedParameters, getSavePath, saveModel= True)

    levelValues = [learningRate,buffersize]
    levelNames = ["learningRate", "buffersize"]
    
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    
    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)
    
    

    plotRowNum = len(learningRate)
    plotColNum = len(buffersize)
    plotCounter = 1
    axs = []
    figure = plt.figure(figsize=(12, 10))
    
    for keyCol, outterSubDf in modelResultDf.groupby('buffersize'):
        for keyRow, innerSubDf in outterSubDf.groupby("learningRate"):
            subplot = figure.add_subplot(plotRowNum, plotColNum, plotCounter)
            axs.append(subplot)
            plotCounter += 1
            plt.ylim([0, 300])
            innerSubDf.T.plot(ax=subplot)
    dirName = os.path.dirname(__file__)
    plotPath = os.path.join(dirName,'plots')
    plt.savefig(os.path.join(plotPath, 'CarpoleEvaluation'))
    plt.show()



if __name__ == "__main__":
    main()
