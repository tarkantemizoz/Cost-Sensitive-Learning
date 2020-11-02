#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
import torch
import numpy as np

from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from bayes_opt.utils import give_set, give_score, test_learning

class cslr_bayes: 

    def __init__(self, data, returns, config):
        
        self.device = config.get("device", "cpu")
        self.cval = config.get("cval", False)
        self.n_val_splits = config.get("n_val_splits", 10)    
        self.rep = config.get("rep", 0) 
        self.set_num = config.get("set", 0) 
        self.n_trials = config.get("n_trials", 2)
        self.val_sets = config.get("val_sets", None)
        self.batchnorm = config.get("batchnorm", False)
        self.n_epochs = config.get("n_epochs", 2000)
        self.n_steps = config.get("n_steps", 1000)        
        self.batches = config.get("batches", None)
        self.batch_size = config.get("batch_size", 1000)
        self.max_batch = config["max_batch"]
        self.layers = config.get("layers", None)        
        self.num_layer = config.get("dnn_layers", 1)    
        self.hidden_size = config.get("hidden_size", [])
        self.lr = config.get("lr", None)   
        self.lr_rate = config.get("lr_rate", 0.001)    
        self.data = data
        self.returns = returns
    
    def train_bayes_linear(self, trial):                 
                  
        space = {'lr_rate' : (trial.suggest_uniform('lr_rate', 0.00005, 0.01) if self.lr is not None else self.lr_rate),
                'batch_size': (trial.suggest_int('batch_size', self.batches[0], self.batches[1]) if self.batches is not None
                               else self.batch_size),
                 'num_layer': (trial.suggest_int('num_layer', self.layers[0], self.layers[1]) if self.layers is not None
                              else self.num_layer)
                }
              
        config_nn = {
            "n_inputs" : self.data.shape[1],
            "dnn_layers" : space['num_layer'],
            "hidden_size" : self.hidden_size,
            "n_outputs" : self.returns.shape[1],
            "batchnorm" : self.batchnorm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps
        }
        
        if self.cval == False:
              
            model = LinearNet(config_nn).to(self.device)
            config_nn["batch_size"] = (2**space['batch_size'] if space['batch_size']<self.max_batch else len(self.data))                     
            optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
            optimization = Optimization(model, optimizer, config_nn)
            optimization.train(self.data, self.returns)
            _, _, test_probs = optimization.evaluate(self.data, self.returns)
            test_return = test_learning(test_probs, returns)
            
        else:
            
            test_return = np.zeros(self.n_val_splits)
            
            if self.val_sets is None:
                skf = KFold(self.n_val_splits, shuffle=True)
                for train_index, test_index in skf.split(self.data):
                    x_train, x_test = self.data[train_index], self.data[test_index]
                    r_train, r_test = self.returns[train_index], self.returns[test_index]
                    model = LinearNet(config_nn).to(self.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
                    config_nn["batch_size"] = (2**space['batch_size'] if space['batch_size']<self.max_batch else len(x_train))               
                    optimization = Optimization(model, optimizer, config_nn)
                    optimization.train(x_train, r_train, x_test, r_test)
                    _, _, test_probs = optimization.evaluate(x_test, r_test)
                    test_return[i],_ = test_learning(test_probs, r_test)            
            
            else:
                for i in range(self.n_val_splits):
                    train_index, test_index = give_set(self.rep, self.set_num, i, self.val_sets)
                    x_train, x_test = self.data[train_index], self.data[test_index]
                    r_train, r_test = self.returns[train_index], self.returns[test_index]
                    model = LinearNet(config_nn).to(self.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
                    config_nn["batch_size"] = (2**space['batch_size'] if space['batch_size']<self.max_batch else len(x_train))               
                    optimization = Optimization(model, optimizer, config_nn)
                    optimization.train(x_train, r_train, x_test, r_test)
                    _, _, test_probs = optimization.evaluate(x_test, r_test)
                    test_return[i],_ = test_learning(test_probs.detach().cpu().clone().numpy(), r_test)
        
        return test_return.mean()

    def bayes(self):

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_linear, n_trials=self.n_trials)
    
        return study.best_params

