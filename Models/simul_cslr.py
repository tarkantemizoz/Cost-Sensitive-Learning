#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import pandas as pd 
import numpy as np
import xgboost as xgb
import torch

from gurobipy import *
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from Models.gurobi_opt import Optimization_MIP
from bayes_opt.utils import test_learning
from Models.cslr import cslr

class data_gen:
    """ A helper class to simulate data for Cost Sensitive Learning"""
    
    def __init__(self, config):
        self.seed = config.get("seed", random.randint(0,100))
        self.n = config.get("n", 2000)
        self.n_test = config.get("n_test", 20000)
        self.noise = config.get("noise", 0)
        self.num_features = config.get("num_features", 10)
        self.num_class = config.get("num_class",2)
        self.size = ("small" if self.num_features < 21 else "large")
        random.seed(self.seed)
        self.x_train = np.zeros((self.n, self.num_features))
        self.x_test = np.zeros((self.n_test, self.num_features))        
        for i in range(self.n):
            for j in range(self.num_features):
                    self.x_train[i,j] = (np.random.normal(0,1) if j % 2 == 0 else random.randint(0, 1))
        for i in range(self.n_test):
            for j in range(self.num_features):
                    self.x_test[i,j] = (np.random.normal(0,1) if j % 2 == 0 else random.randint(0, 1))                    
        self.train_y = self.decision(self.x_train)
        self.test_y = self.decision(self.x_test)
        mean = self.train_y.mean()
        std = self.train_y.std()
        self.train_y = (self.train_y - mean) / std
        self.test_y = (self.test_y - mean) / std
        for i in range(len(self.train_y)):
            for j in range(self.num_class):
                self.train_y[i,j] = self.train_y[i,j] + np.random.normal(0,self.noise)
        self.r_train, self.outcomes_train = self.returns(self.x_train, self.train_y)
        self.r_test, self.outcomes_test =  self.returns(self.x_test, self.test_y)
        self.rmax_train = np.zeros(len(self.r_train))
        for i in range(len(self.r_train)):
            self.rmax_train[i] = self.r_train[i,np.argmax(self.r_train[i])]
        self.rmax_test = np.zeros(len(self.r_test))
        for i in range(len(self.r_test)):
            self.rmax_test[i] = self.r_test[i,np.argmax(self.r_test[i])]
            
    def returns(self, arr, y):        
        returns = np.zeros((len(arr),self.num_class))
        outcomes = np.zeros(len(arr))
        if self.num_class == 2:   
            for i in range(len(arr)):
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(100,500) if np.argmax(y[i]) == 0 else -random.uniform(50,150))    
                returns[i,1] = (random.uniform(500,2000) if np.argmax(y[i]) == 1 else -random.uniform(50,150))    
        else:
            for i in range(len(arr)):
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(500,2000) if np.argmax(y[i]) == 0 else -random.uniform(50,150))
                returns[i,1] = (random.uniform(300,500) if np.argmax(y[i]) == 1 else -random.uniform(50,150))
                returns[i,2] = (random.uniform(100,300) if np.argmax(y[i]) == 2 else -random.uniform(50,150))               
        return returns, outcomes.astype(int)     
    
    def decision(self, arr):        
        y = np.zeros((len(arr),self.num_class))
        if self.num_class == 2:
            if self.size == "small":
                for i in range(len(arr)):                   
                    y[i,0] = sum(arr[i,(0,1,7,10,13)]) + (np.where(arr[i,6] > -0.6, 5, 0) + abs(arr[i,2] * arr[i,4]))
                    y[i,1] = sum(arr[i,(0,1,7,10,13)])+ abs(arr[i,12] * arr[i,14])                  
            else:
                for i in range(len(arr)):
                    y[i,0] = sum(arr[i,(0,1,6,9,11,16,23)]) + (np.where(arr[i,10] > -0.6, 5, 0) +
                                                                abs(arr[i,18] * arr[i,20]) + arr[i,23])
                    y[i,1] = sum(arr[i,(0,1,6,9,11,16,23)]) + abs(arr[i,12] * arr[i,14]) + arr[i,19]      
        else:
            if self.size == "small":
                for i in range(len(arr)):
                    y[i,0] = sum(arr[i,(0,1,7,10,13)]) + 2
                    y[i,1] = sum(arr[i,(0,1,7,10,13)]) + (np.where(arr[i,8] > 0, 7, 0) + abs(arr[i,0] * arr[i,12]))
                    y[i,2] = sum(arr[i,(0,1,7,10,13)]) + (np.where(arr[i,6] > -0.2, 7, 0) + abs(arr[i,2] * arr[i,14]))    
            else:
                for i in range(len(arr)):
                    y[i,0] = sum(arr[i,(0,1,6,9,11,16,23)])  + 2
                    y[i,1] = sum(arr[i,(0,1,6,9,11,16,23)]) + (np.where(arr[i,22] > 0, 7, 0) + abs(arr[i,2] * arr[i,14]) + arr[i,21])
                    y[i,2] = sum(arr[i,(0,1,6,9,11,16,23)]) + (np.where(arr[i,10] > -0.2, 7, 0) + abs(arr[i,12] * arr[i,20]) + arr[i,19])                
        return y        
           

class cs_synthetic(data_gen): 

    def __init__(self, config):
        super(cs_synthetic, self).__init__(config)
        self.device = config.get("device", "cpu")
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", 500)    
        self.lr_rate = config.get("lr_rate", 5e-3)    
        self.xgsteps = config.get("xgsteps", 20)
        self.batchnorm = config.get("batchnorm", False)
        self.batch_size = config.get("batch_size", 1000)
        self.time_limit = config.get("time_limit", 100.0)
        self.give_initial = config.get("initial", False)
        self.beta_initial = None
        self.model = None
    
    def gradient_simul(self):
       
        config_nn = {
            "lr_rate": self.lr_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "batchnorm": self.batchnorm
        }               
        self.model = cslr(config_nn,
                     self.x_train,
                     self.r_train,
                     self.x_test,
                     self.r_test)
        test_probs, train_probs = self.model.gradient()
        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, _ = test_learning(train_probs, self.r_train)              
        return (test_return/sum(self.rmax_test), train_return/sum(self.rmax_train),
                accuracy_score(self.outcomes_test, test_outcome),
                f1_score(self.outcomes_test, test_outcome, average="weighted"))
    
    def mip_opt_simul(self):

        if self.give_initial == True:
            if self.model is None:
                self.gradient_simul()
            self.model.give_initial = True
            self.model.time_limit = self.time_limit
        else:
            if self.model is None:
                config_mip = {
                    "give_initial": False,
                    "time_limit": self.time_limit
                }
                self.model = cslr(config_mip,
                             self.x_train,
                             self.r_train,
                             self.x_test,
                             self.r_test)
            else:
                self.model.time_limit = self.time_limit
                self.model.give_initial = False
        test_probs, train_probs = self.model.mip_opt()
        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, _ = test_learning(train_probs, self.r_train)            
        return (test_return/sum(self.rmax_test), train_return/sum(self.rmax_train),
                accuracy_score(self.outcomes_test, test_outcome),
                f1_score(self.outcomes_test, test_outcome, average="weighted"))       

    def tree(self):
        
        probs = DecisionTreeClassifier().fit(self.x_train, self.outcomes_train).predict_proba(self.x_test)
        test_return, test_outcome = test_learning(probs, self.r_test)
        return (test_return/sum(self.rmax_test),
                accuracy_score(self.outcomes_test, test_outcome),
                f1_score(self.outcomes_test, test_outcome, average="weighted"))
    
    def logistic(self):
        
        probs = LogisticRegression().fit(self.x_train, self.outcomes_train).predict_proba(self.x_test) 
        test_return, test_outcome = test_learning(probs, self.r_test)
        return (test_return/sum(self.rmax_test),
                accuracy_score(self.outcomes_test, test_outcome),
                f1_score(self.outcomes_test, test_outcome, average="weighted"))

    def xgboost(self):
        
        dtrain = xgb.DMatrix(self.x_train, label=self.outcomes_train)
        dtest = xgb.DMatrix(self.x_test, label=self.outcomes_test)  
        param = {'objective': 'multi:softprob', 'num_class': self.num_class}
        probs = xgb.train(param, dtrain, self.xgsteps).predict(dtest)
        test_return, test_outcome = test_learning(probs, self.r_test)
        return (test_return/sum(self.rmax_test),
                accuracy_score(self.outcomes_test, test_outcome),
                f1_score(self.outcomes_test, test_outcome, average="weighted"))
           
        

