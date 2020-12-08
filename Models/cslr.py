#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import math

from sklearn.metrics import f1_score, accuracy_score

from bayes_opt.bayes_linearnet import cslr_bayes
from bayes_opt.utils import test_learning
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
#from Models.gurobi_opt import Optimization_MIP
#from gurobipy import *


class cslr: 

    def __init__(self,
                 formatter,
                 train,
                 test,
                 valid=None
                ):

        self.formatter = formatter
        self.fixed_params = self.formatter.get_experiment_params()
        self.expt_params = self.formatter.get_default_model_params()
        self.beta_initial = None   
        
        self.x_train = train[0]
        self.x_test = test[0] 
        self.r_train = train[1]
        self.r_test = test[1]
        self.rmax_train = sum(train[2])        
        self.rmax_test = sum(test[2]) 
        self.y_train = train[3]     
        self.y_test = test[3]        
        self.x_val = (valid[0] if valid is not None and valid != [] else None)
        self.r_val = (valid[1] if valid is not None and valid != [] else None)
        self.rmax_val = (valid[2] if valid is not None and valid != [] else None)   
        self.y_val = (valid[3] if valid is not None and valid != [] else None)
        self.train_scores = []
        self.test_scores = []
        self.val_scores = []
                                      
    def opt_initial(self, beta_opt):
        
        scores = np.zeros((len(self.x_train), self.num_class))
        scores_diff = np.zeros((len(self.x_train),
                                ((self.num_class * (self.num_class - 1)) // 2)))
        for i in range(len(self.x_train)):
            for k in range(self.num_class):
                scores[i,k] = sum(self.x_train[i,j] * beta_opt[k,j].item()
                                  for j in range(self.num_features))
        diff_ind = 0
        for k in range(self.num_class-1):
            for t in range(self.num_class-(k+1)):
                scores_diff[:,diff_ind] = np.subtract(scores[:,k], scores[:,(k+t+1)])
                diff_ind += 1
        is_deviate = []
        for i in range(len(scores_diff)):
            if (abs(scores_diff[i,:]) <= 0.01).any() or (scores[i,:] >= 100).any():
                is_deviate.append(i)
        self.x_train = np.delete(self.x_train, is_deviate, 0)
        self.r_train = np.delete(self.r_train, is_deviate, 0)
        self.rmax_train = np.delete(self.rmax_train, is_deviate, 0)
        self.y_train = np.delete(self.x_train, is_deviate, 0)  
    
    def gradient(self):

        self.dnn_layers = int(self.expt_params.get("dnn_layers", 1))
        self.hidden_size = self.expt_params.get("hidden_size", [])
        self.lr_rate = float(self.expt_params.get("learning_rate", 5e-3))   
        self.batch_size = int(self.expt_params.get("batch_size", len(self.x_train)))
        self.max_batch = math.floor(math.log2(self.x_train.shape[0]))  
        self.bayes = self.formatter.bayes
        self.model = self.expt_params.get("model", None)     
        
        self.scaler = self.fixed_params.get("scaler", False)        
        self.device = str(self.fixed_params.get("device", "cpu"))
        self.n_epochs = int(self.fixed_params.get("n_epochs", 1000))
        self.n_steps = int(self.fixed_params.get("n_steps", self.n_epochs))    
        self.batch_norm = self.expt_params.get("batch_norm", False)        
        self.num_class = self.r_train.shape[1]
        self.num_features = self.x_train.shape[1]                     
        
        params = {
            "n_inputs" : self.num_features,
            "n_outputs": self.num_class,
            "batch_norm": self.batch_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "device": self.device,
            "scaler": self.scaler
            }        
        
        if self.bayes == True:
            
            bayes = cslr_bayes(self.formatter, self.x_train, self.r_train, params)
            bayes_params = self.formatter.get_bayes_params()
            best_params_bayes = bayes.bayes()
            best_lr = (best_params_bayes.get("lr_rate", ""))
            best_batch_size = (best_params_bayes.get("batch_size" "")
                               if bayes_params["batch_size_bayes"] is not None
                               else self.max_batch)
            best_dnn_layers = (best_params_bayes.get("dnn_layers" "")
                          if bayes_params["dnn_layers_bayes"] is not None
                               else self.dnn_layers)
            params["dnn_layers"] = best_dnn_layers         
            params["batch_size"] = (2**best_batch_size
                                       if best_batch_size<self.max_batch
                                    else len(self.x_train))
        else:
            best_lr = self.lr_rate
            params["batch_size"] = self.batch_size
            params["dnn_layers"] = self.dnn_layers
        
        if self.scaler == True:
            if self.x_val is not None:
                self.x_train, self.x_test, self.x_val = self.formatter.transform_inputs(self.x_train,
                                                                            self.x_test,
                                                                            self.x_val)
            else:
                self.x_train, self.x_test = self.formatter.transform_inputs(self.x_train,
                                                                self.x_test)  
                
        if len(self.hidden_size) != params["dnn_layers"]:
            for _ in range((params["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor(math.sqrt(self.num_features + self.num_class) + 5))
        params["hidden_size"] = self.hidden_size
        
        x_train_nn = torch.Tensor(self.x_train).to(self.device)   
        x_test_nn = torch.Tensor(self.x_test).to(self.device)
        r_train_nn = torch.Tensor(self.r_train).to(self.device)
        r_test_nn = torch.Tensor(self.r_test).to(self.device)            
                        
        if self.model is None:
            self.model = LinearNet(params).to(self.device)
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_lr)
        optimization = Optimization(self.model, optimizer, params)  
        
        if self.x_val is not None:
            x_val_nn = torch.Tensor(self.x_val).to(self.device)
            r_val_nn = torch.Tensor(self.r_val).to(self.device)
            optimization.train(x_train_nn, r_train_nn, x_val_nn, r_val_nn)
            _, _, val_probs = optimization.evaluate(x_val_nn, r_val_nn)
            val_probs = val_probs.detach().cpu().clone().numpy()             
        else:
            optimization.train(x_train_nn, r_train_nn, x_test_nn, r_test_nn)                
        _, _, test_probs = optimization.evaluate(x_test_nn, r_test_nn)
        _, _, train_probs = optimization.evaluate(x_train_nn, r_train_nn)    
        test_probs = test_probs.detach().cpu().clone().numpy()
        train_probs = train_probs.detach().cpu().clone().numpy() 
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "layer" in name:
                    self.beta_initial = param.data.detach().cpu().clone().numpy() 
                    
        torch.save(self.model.state_dict(), self.formatter.model_path+"_cslr.pt")                     
        
        return (self.return_results(train_probs, test_probs, val_probs)
                if self.x_val is not None
                else (self.return_results(train_probs, test_probs)))
    
    def ret_model(self):
        return self.model
    
    def mip_opt(self):
    
        if self.give_initial == True:
            model_name = "mip_wi"
            if self.beta_initial is None:
                self.gradient()
            self.opt_initial(self.beta_initial)
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
            for i in range(self.num_class):
                for j in range(self.num_features):
                    self.model_mip.beta[i,j].start = self.beta_initial[i,j]
            self.model_mip.m.update() 
        else:
            model_name = "mip"            
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
        path = self.formatter.model_path+"_"+model_name
        self.model_mip.m.modelSense = GRB.MAXIMIZE
        self.model_mip.m.setParam(GRB.Param.TimeLimit, self.time_limit)
        #self.model_mip.m.write(path+'.lp')        
        self.model_mip.m.optimize()       
        variables = self.model_mip.m.getVars()
        #pickle.dump(variables, open(path+"_params.dat", "wb"))
        
        test_probs = np.zeros((len(self.x_test), self.num_class))
        train_probs = np.zeros((len(self.x_train), self.num_class))  
        
        try:
            for i in range(len(self.x_test)):
                for k in range(self.num_class):
                    test_probs[i,k] = sum(self.x_test[i,j] * self.model_mip.beta[k,j].x
                                          for j in range(self.num_features))
            for i in range(len(self.x_train)):
                for k in range(self.num_class):
                    train_probs[i,k] = sum(self.x_train[i,j] * self.model_mip.beta[k,j].x
                                           for j in range(self.num_features))  
            if self.x_val is not None:
                val_probs = np.zeros((len(self.x_val), self.num_class))       
                for i in range(len(self.x_val)):
                    for k in range(self.num_class):
                        val_probs[i,k] = sum(self.x_val[i,j] * self.model_mip.beta[k,j].x
                                             for j in range(self.num_features))  
            return (self.return_results(train_probs, test_probs, val_probs)
                    if self.x_val is not None
                    else (self.return_results(train_probs, test_probs)))
        except:
            return (self.return_results(train_probs, test_probs, val_probs)
                    if self.x_val is not None
                    else (self.return_results(train_probs, test_probs)))
            
     
    def return_results(self, train_probs, test_probs, val_probs=None):
        
        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, train_outcome = test_learning(train_probs, self.r_train)
        test_return = test_return / self.rmax_test
        train_return = train_return / self.rmax_train 
        test_acc, test_f1 = (accuracy_score(self.y_test, test_outcome),
                             f1_score(self.y_test, test_outcome, average="weighted")
                            )   
        train_acc, train_f1 = (accuracy_score(self.y_train, train_outcome),
                               f1_score(self.y_train, train_outcome, average="weighted")
                              )  
        self.test_scores, self.train_scores = ([test_return, test_acc, test_f1],
                                     [train_return, train_acc, train_f1]
                                    )
        
        if val_probs is not None:
            
            val_return, val_outcome = test_learning(val_probs, self.r_val)
            val_return = val_return / self.rmax_val
            val_acc, val_f1 = (accuracy_score(self.y_val, val_outcome),
                                 f1_score(self.y_val, val_outcome, average="weighted"))  
            self.val_scores = [val_return, val_acc, val_f1]
            
        return self.train_scores, self.test_scores, self.val_scores
        

