# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:14:08 2020

@author: martin
"""


import os
import sys
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.DDPG import Agent
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from collections import deque, OrderedDict


def main():
    Env_name = 'Walker2d-v1'
    env = gym.make(Env_name)
    statedim = env.observation_space.shape[0]
    actiondim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2
    
    fixedParameters = OrderedDict()
    fixedParameters['statedim'] = statedim
    fixedParameters['actiondim'] = actiondim
    fixedParameters['actionHigh'] = actionHigh
    fixedParameters['actionLow'] = actionLow
    fixedParameters['actionBound'] = actionBound
    fixedParameters['initweight'] = tf.random_normal_initializer(0., 0.1)
    fixedParameters['initbias'] = tf.constant_initializer(0.1)
    fixedParameters['gamma'] = 0.99
    fixedParameters['tau'] = 0.001
    fixedParameters['buffersize'] = 500000
    fixedParameters['noiseDacayStep'] = fixedParameters['buffersize']
    fixedParameters['initbuffer'] = 10000      
    fixedParameters['batchsize'] = 64
    fixedParameters['minVar'] = 0.001
    fixedParameters['actorlearningRate'] = 0.0001
    fixedParameters['criticlearningRate'] = 0.001
    fixedParameters['actornumberlayers'] = 256
    fixedParameters['criticnumberlayers'] = 256
    fixedParameters['noiseDecay'] = 0.00001
    fixedParameters['initnoisevar'] = 2
    fixedParameters['EPISODE'] = 2000
    fixedParameters['maxTimeStep'] = 1000
    
    agent = Agent(fixedParameters)
    agent(env)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    