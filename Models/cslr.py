#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np

from bayes_opt.bayes_linearnet import cslr_bayes
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from Models.gurobi_opt import Optimization_MIP
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gurobipy import *


class cslr: 

    def __init__(self, config, x_train, r_train, x_test, r_test, x_val=None, r_val=None):
        self.device = config.get("device", "cpu")
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", "500")    
        self.lr_rate = config.get("lr_rate", 5e-3)   
        self.hidden_size = config.get("hidden_size", 0)
        self.batch_size = config.get("batch_size", 1000)
        self.num_layer = config.get("numlayer", 1)
        self.batchnorm = config.get("batchnorm", False)
        self.scaler = config.get("scaler", None)
        self.batches_bayes = config.get("batches", None)
        self.lr_bayes = config.get("lr", None)
        self.layers_bayes = config.get("layers", None)
        self.bayes = config.get("bayes", False)
        self.rep = config.get("rep", None)
        self.set = config.get("set", None)
        self.cval = config.get("cval", None)
        self.n_val_splits = config.get("n_val_splits", 10)
        self.val_sets = config.get("val_sets", None)
        self.n_trials = config.get("n_trials", None)   
        self.time_limit = config.get("time_limit", 100.0)
        self.give_initial = config.get("initial", False)        
        self.beta_initial = None
        self.x_train = x_train
        self.x_test = x_test
        self.r_train = r_train
        self.r_test = r_test
        self.x_val = x_val
        self.r_val = r_val
        self.max_batch = config.get("max_batch", len(self.x_train))
        self.num_class = r_train.shape[1]
        self.num_features = x_train.shape[1]
        self.model = config.get("model", None)        
        
    def opt_initial(self, beta_opt):
        
        scores = np.zeros((len(self.x_train), self.num_class))
        scores_diff = np.zeros((len(self.x_train), ((self.num_class * (self.num_class - 1)) // 2)))
        for i in range(len(self.x_train)):
            for k in range(self.num_class):
                scores[i,k] = sum(self.x_train[i,j] * beta_opt[k,j].item() for j in range(self.num_features))
        diff_ind = 0
        for k in range(self.num_class-1):
            for t in range(self.num_class-(k+1)):
                scores_diff[:,diff_ind] = np.subtract(scores[:,k], scores[:,(k+t+1)])
                diff_ind += 1
        is_deviate = []
        for i in range(len(scores_diff)):
            if (abs(scores_diff[i,:]) < 0.01).any() or (scores[i,:] > 100).any():
                is_deviate.append(i)
        return  np.delete(self.x_train, is_deviate, 0), np.delete(self.r_train, is_deviate, 0)     
    
    def gradient(self):
        
        if self.scaler is not None:
            scaler = self.scaler
            self.x_train = scaler.fit_transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
            if self.x_val is not None:
                self.x_val = scaler.transform(self.x_val)
        x_train_nn = torch.Tensor(self.x_train).to(self.device)   
        x_test_nn = torch.Tensor(self.x_test).to(self.device)
        r_train_nn = torch.Tensor(self.r_train).to(self.device)
        r_test_nn = torch.Tensor(self.r_test).to(self.device)
        config_nn = {
            "n_inputs" : self.num_features,
            "n_outputs": self.num_class,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "batchnorm": self.batchnorm,
            "hidden_size": self.hidden_size,
            "max_batch": self.max_batch,
            "device": self.device
            }        
        if self.bayes == True:
            config_nn["batches"], config_nn["lr"], config_nn["layers"]  = self.batches_bayes, self.lr_bayes, self.layers_bayes
            config_nn["rep"], config_nn["set"], config_nn["n_val_splits"]  = self.rep, self.set, self.n_val_splits
            config_nn["cval"], config_nn["val_sets"], config_nn["n_trials"] = self.cval, self.val_sets, self.n_trials      
            bayes = cslr_bayes(x_train_nn, r_train_nn, config_nn)
            best_params_bayes = bayes.bayes()
            best_lr = (best_params_bayes.get("lr_rate", "") if self.lr_bayes is not None else self.lr_rate)
            best_batch_size = (best_params_bayes.get("batch_size" "") if self.batches_bayes is not None else self.batch_size)
            best_layer = (best_params_bayes.get("num_layer" "") if self.layers_bayes is not None else self.num_layer)
            config_nn["dnn_layers"] = best_layer         
            config_nn["batch_size"] = (2**best_batch_size if best_batch_size<self.max_batch else len(x_train_nn))
        else:
            best_lr = self.lr_rate
            config_nn["batch_size"] = self.batch_size
            config_nn["dnn_layers"] = self.num_layer
        if self.model is None:
            self.model = LinearNet(config_nn).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_lr)
        optimization = Optimization(self.model, optimizer, config_nn)  
        optimization.train(x_train_nn, r_train_nn, x_test_nn, r_test_nn)
        _, _, test_probs = optimization.evaluate(x_test_nn, r_test_nn)
        _, _, train_probs = optimization.evaluate(x_train_nn, r_train_nn)
        if self.x_val is not None:
            x_val_nn = torch.Tensor(self.x_val).to(self.device)
            r_val_nn = torch.Tensor(self.r_val).to(self.device)                
            _, _, val_probs = optimization.evaluate(x_val_nn, r_val_nn)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "layer" in name:
                    self.beta_initial = param.data.detach().cpu().clone().numpy()        
        return((test_probs.detach().cpu().clone().numpy(),
                train_probs.detach().cpu().clone().numpy(),
                val_probs.detach().cpu().clone().numpy())
               if self.x_val is not None else
               (test_probs.detach().cpu().clone().numpy(),
                train_probs.detach().cpu().clone().numpy()))
    
    def ret_model(self):
        return self.model
    
    def mip_opt(self):
    
        if self.give_initial == True:
            if self.beta_initial is None:
                self.gradient()
            x_train, r_train = self.opt_initial(self.beta_initial)
            self.model_mip = Optimization_MIP({}, x_train, r_train)
            for i in range(self.num_class):
                for j in range(self.num_features):
                    self.model_mip.beta[i,j].start = self.beta_initial[i,j]
            self.model_mip.m.update() 
        else:
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
        self.model_mip.m.modelSense = GRB.MAXIMIZE
        self.model_mip.m.setParam(GRB.Param.TimeLimit, self.time_limit)
        self.model_mip.m.optimize()
        test_probs = np.zeros((len(self.x_test), self.num_class))
        train_probs = np.zeros((len(self.x_train), self.num_class))        
        try:
            for i in range(len(self.x_test)):
                for k in range(self.num_class):
                    test_probs[i,k] = sum(self.x_test[i,j] * self.model_mip.beta[k,j].x for j in range(self.num_features))
            for i in range(len(self.x_train)):
                for k in range(self.num_class):
                    train_probs[i,k] = sum(self.x_train[i,j] * self.model_mip.beta[k,j].x for j in range(self.num_features))  
            if self.x_val is not None:
                val_probs = np.zeros((len(self.x_val), self.num_class))       
                for i in range(len(self.x_val)):
                    for k in range(self.num_class):
                        val_probs[i,k] = sum(self.x_val[i,j] * self.model_mip.beta[k,j].x for j in range(self.num_features))  
            return((test_probs, train_probs, val_probs) if self.x_val is not None else (test_probs, train_probs))
        except:
             return((test_probs, train_probs, val_probs) if self.x_val is not None else (test_probs, train_probs))

