#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
import torch
import numpy as np
import math

from sklearn.model_selection import KFold
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from bayes_opt.utils import test_learning

class cslr_bayes: 

    def __init__(self, formatter, data, returns, config):

        self.device = config.get("device", "cpu")
        self.scaler = config.get("scaler", False)          
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", self.n_epochs)        
        self.batch_norm = config.get("batch_norm", False)        
        self.batch_size = config.get("batch_size", len(data))
        self.dnn_layers = config.get("dnn_layers", 1)    
        self.hidden_size = config.get("hidden_size", [])
        
        self.formatter = formatter      
        self.bayes_params = self.formatter.get_bayes_params()
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.inner_cval = self.bayes_params["inner_cval"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10)            
        self.batch_size_bayes = self.bayes_params["batch_size_bayes"]
        self.dnn_layers_bayes = self.bayes_params["dnn_layers_bayes"]
        
        self.data = data
        self.returns = returns
        self.max_batch = math.floor(math.log2(self.data.shape[0]))     
    
    def train_bayes_linear(self, trial):                 
                  
        space = {'lr_rate' : (trial.suggest_uniform('lr_rate', 0.00005, 0.01)),
                'batch_size': (trial.suggest_int('batch_size',
                                                 self.batch_size_bayes[0],
                                                 self.batch_size_bayes[1])
                               if self.batch_size_bayes is not None
                               else self.max_batch),
                 'dnn_layers': (trial.suggest_int('dnn_layers',
                                                  self.dnn_layers_bayes[0],
                                                  self.dnn_layers_bayes[1])
                               if self.dnn_layers_bayes is not None
                              else self.dnn_layers)
                }
              
        config_nn = {
            "n_inputs" : self.data.shape[1],
            "dnn_layers" : space['dnn_layers'],
            "n_outputs" : self.returns.shape[1],
            "batch_norm" : self.batch_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps
        }

        if len(self.hidden_size) != config_nn["dnn_layers"]:
            for _ in range((config_nn["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor
                                        (math.sqrt(config_nn["n_inputs"]
                                                   + config_nn["n_outputs"]) + 5
                                        )
                                       )
        config_nn["hidden_size"] = self.hidden_size
        
        if self.inner_cval == False:
            
            x = self.data
            if self.scaler == True:
                x = self.formatter.transform_inputs(x)            
                
            x_nn = torch.Tensor(x).to(self.device)   
            r_nn = torch.Tensor(self.returns).to(self.device)
                       
            model = LinearNet(config_nn).to(self.device)
            config_nn["batch_size"] = (2**space['batch_size']
                                       if space['batch_size']<self.max_batch
                                       else len(x))                     
            optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
            optimization = Optimization(model, optimizer, config_nn)
            optimization.train(x_nn, r_nn)
            _, _, test_probs = optimization.evaluate(x_nn, r_nn)
            result,_ = test_learning(test_probs.detach().cpu().clone().numpy(),
                                        self.returns)
            
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = KFold(self.n_val_splits, shuffle=True)
            
            for train_index, test_index in skf.split(self.data):
                
                x_train, x_test = self.data[train_index], self.data[test_index]
                r_train, r_test = self.returns[train_index], self.returns[test_index]
                                
                if self.scaler == True:
                    x_train, x_test = self.formatter.transform_inputs(x_train, x_test)

                x_train = torch.Tensor(x_train).to(self.device)   
                x_test = torch.Tensor(x_test).to(self.device)
                r_train = torch.Tensor(r_train).to(self.device)
                r_test = torch.Tensor(r_test).to(self.device)                 
                        
                model = LinearNet(config_nn).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
                self.max_batch = math.floor(math.log2(x_train.shape[0]))                
                config_nn["batch_size"] = (2**space['batch_size']
                                           if space['batch_size']<self.max_batch
                                           else len(x_train))               
                optimization = Optimization(model, optimizer, config_nn)
                optimization.train(x_train, r_train, x_test, r_test)
                _, _, test_probs = optimization.evaluate(x_test, r_test)
                test_return[count],_ = test_learning(test_probs.detach().cpu().clone().numpy(),
                                                     r_test)
                count += 1
                
            result = np.mean(test_return)
                
        return result
    
    def bayes(self):

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_linear, n_trials=self.bayes_trials)
    
        return study.best_params

