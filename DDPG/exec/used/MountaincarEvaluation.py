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
from env.MountainCarDiscrete import MtCarDiscreteEnvSetup,visualizeMtCarDiscrete,resetMtCarDiscrete,MtCarDiscreteTransition,MtCarDiscreteReward,MtCarDiscreteIsTerminal
os.environ['KMP_DUPLICATE_LIB_OK']='True'



ENV_NAME = 'MountainCar-v0'
env = MtCarDiscreteEnvSetup()
step = 200

class EvaluateLearningrateAndbatchsize:
    def __init__(self, fixedParameters, getSavePath, saveModel = True):
        self.fixedParameters = fixedParameters
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        batchsize = df.index.get_level_values('batchsize')[0]
        buffersize = df.index.get_level_values('buffersize')[0]
        
            
        visualize = visualizeMtCarDiscrete() 
        reset = resetMtCarDiscrete(1234)         
        transition = MtCarDiscreteTransition()  
        rewardMtCar = MtCarDiscreteReward()        
        isterminal = MtCarDiscreteIsTerminal()
        replaybuffer = deque(maxlen=int(buffersize))
        trajectory = []
        totalrewards = []
        totalreward = []
        averagerewards = []
        
        buildmodel = BuildModel(self.fixedParameters['stateDim'],self.fixedParameters['actionDim'])
        Writer,DQNmodel = buildmodel(self.fixedParameters['numberlayers'])
        replaceParameters = ReplaceParameters(self.fixedParameters['replaceiter'])
        trainModel = TrainModel(self.fixedParameters['learningRate'], self.fixedParameters['gamma'],Writer)
        trainDQNmodel = TrainDQNmodel(replaceParameters, trainModel, DQNmodel)
        learn = Learn(buffersize,batchsize,trainDQNmodel,self.fixedParameters['actionDim'])
        runepsilon = self.fixedParameters['initepsilon']
        
        
        
        for episode in range(self.fixedParameters['maxEpisode']):
            state  = reset()
            rewards = 0
            runtime = 0
            done = False
            while True:
                action = learn.Getaction(DQNmodel,runepsilon,state)  
                nextstate=transition(state, action)
                done = isterminal(nextstate)
                reward = rewardMtCar(state,action,nextstate,done)
                trajectory.append((state, action, reward, nextstate))
                learn.ReplayMemory(replaybuffer,state, action, reward, nextstate,done)
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
            runepsilon = epsilonDec(runepsilon,self.fixedParameters['minepsilon'],self.fixedParameters['epsilondec'])
            if episode % 100 == 0:
                averagerewards.append(np.mean(totalreward))
                print('episode:',episode,'meanreward:',np.mean(totalreward))
                totalreward = []




        timeStep = 100*(np.arange(len(averagerewards)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, averagerewards)})

        if self.saveModel:
            Parameters = {'gamma': gamma, 'buffersize': buffersize }
            modelPath = self.getSavePath(Parameters)
            with DQNmodel.as_default():
                saveVariables(DQNmodel, modelPath)

        return resultSe




def main():
        
    statesdim = env.observation_space.shape[0]
    actiondim = env.action_space.n
    
    fixedParameters = OrderedDict()
    fixedParameters['learningRate'] = 0.001
    fixedParameters['maxEpisode'] = 3001
    fixedParameters['gamma'] = 0.995
    fixedParameters['numberlayers'] = 256      
    fixedParameters['stateDim'] = statesdim
    fixedParameters['actionDim'] = actiondim
    fixedParameters['replaceiter'] = 100
    fixedParameters['initepsilon'] = 1
    fixedParameters['minepsilon'] = 0.01
    fixedParameters['epsilondec'] = 0.001


    batchsize = [128,256,512]
    buffersize = [10000,20000,50000]

    modelSaveDirectory = "/path/to/logs/trainedDQNModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    evaluateModel = EvaluateLearningrateAndbatchsize(fixedParameters, getSavePath, saveModel= False)

    levelValues = [batchsize,buffersize]
    levelNames = ["batchsize", "buffersize"]
    
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    
    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)
    
    
    plotColNum = len(buffersize)
    plotRowNum = len(batchsize)
    plotCounter = 1
    axs = []
    figure = plt.figure(figsize=(12, 10))
    
    for keyCol, outterSubDf in modelResultDf.groupby('buffersize'):
        for keyRow, innerSubDf in outterSubDf.groupby("batchsize"):
            subplot = figure.add_subplot(plotRowNum, plotColNum, plotCounter)
            axs.append(subplot)
            plotCounter += 1
            plt.ylim([-200,-70])
            innerSubDf.T.plot(ax=subplot)
    dirName = os.path.dirname(__file__)
    plotPath = os.path.join(dirName,'plots')
    plt.savefig(os.path.join(plotPath, 'MountaincarEvaluation'))
    plt.show()



if __name__ == "__main__":
    main()


