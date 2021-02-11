# coding: utf-8
# Copyright 2020 Tarkan Temizoz

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import math
import pickle

from sklearn.metrics import f1_score, accuracy_score
from gurobipy import *

from bayes_opt.bayes_linearnet import cslr_bayes
from bayes_opt.utils import test_learning
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from Models.gurobi_opt import Optimization_MIP

class cslr: 
    """A class to run the proposed EDCS algorithms: CSLR, MIP-WI and MIP
        
    Attributes:
        formatter: formatter for the specified experiment
        fixed_params: fixed parameters for the specified experiment
        expt_params: experiment parameters for CSLR
        expt: name of EDCS learning model
        time_limit: time limit for MIP-WI and MIP
        beta_initial: initial coefficients to be fed into MIP model
        analyze_first_feasible: whether to find the first feasible solution
            for MIP or MIP-WI
        first_feasible_continue: whether to continue optimizing after finding the
            first feasible solution
        train_scores: a triple of: (normalized return, accuracy, weighted F score)
            for train data
        test_scores: a triple of: (normalized return, accuracy, weighted F score)
            for test data
        val_scores: a triple of: (normalized return, accuracy, weighted F score)
            for validation data
        num_class: number of classes of an instance
        num_features: number of features of an instance
        x_train: features for train set
        r_train: returns for train set
        rmax_train: maximum returns for train set
        y_train: labels for train set
        The same applies for test and validation sets.
    """
    
    def __init__(self,
                 formatter,
                 train,
                 test,
                 valid=None
                ):
        """Organizes the data for CSLR, MIP and MIP-WI
            
        Args:
            formatter: formatter for the specified experiment
            train: holds the quadruple of: (features, returns, maximum returns, labels)
                for train data
            test: holds the quadruple of: (features, returns, maximum returns, labels)
                for test data
            valid: holds the quadruple of: (features, returns, maximum returns, labels)
                for validation data
        """
        
        self.formatter = formatter
        self.fixed_params = self.formatter.get_experiment_params()
        self.expt_params = self.formatter.get_default_model_params()
        self.time_limit = self.formatter.time_limit
        self.beta_initial = None
        self.analyze_first_feasible = True
        self.first_feasible_continue = False
        self.expt = "cslr"
        
        self.x_train = train[0]
        self.x_test = test[0] 
        self.r_train = train[1]
        self.r_test = test[1]
        self.rmax_train = train[2]       
        self.rmax_test = test[2]
        self.y_train = train[3]     
        self.y_test = test[3]        
        self.x_val = (valid[0] if valid is not None and valid != [] else None)
        self.r_val = (valid[1] if valid is not None and valid != [] else None)
        self.rmax_val = (valid[2] if valid is not None and valid != [] else None)   
        self.y_val = (valid[3] if valid is not None and valid != [] else None)
        self.train_scores = []
        self.test_scores = []
        self.val_scores = []
        self.num_class = self.r_train.shape[1]
        self.num_features = self.x_train.shape[1]
        
    def opt_initial(self, beta_opt):
        """Removing instances that creates infeasibility for mip-wi"""
        
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
    
        # finding the infeasible instances
        is_deviate = []
        for i in range(len(scores_diff)):
            if (abs(scores_diff[i,:]) <= 0.01).any() or (abs(scores[i,:]) >= 100).any():
                is_deviate.append(i)
        
        # update the input data
        self.x_train = np.delete(self.x_train, is_deviate, 0)
        self.r_train = np.delete(self.r_train, is_deviate, 0)
        self.rmax_train = np.delete(self.rmax_train, is_deviate)
        self.y_train = np.delete(self.y_train, is_deviate)  

    def gradient(self):
        """Running Cost-sensitive Logistic Regression
            
        Returns:
            The function that calculates the results.
        """
        
        # setting up the model parameters
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
        
        params = {
            "n_inputs" : self.num_features,
            "n_outputs": self.num_class,
            "batch_norm": self.batch_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "device": self.device,
            "scaler": self.scaler
            }        
        
        # applying bayesian optimization to find best parameters
        if self.bayes == True:
            
            bayes = cslr_bayes(self.formatter, self.x_train, self.y_train, self.r_train, params)
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
        # continuing with default parameters
        else:
            best_lr = self.lr_rate
            params["batch_size"] = self.batch_size
            params["dnn_layers"] = self.dnn_layers
        
        # calling the prespecified scaler, which depends on the experiment
        if self.scaler == True:
            if self.x_val is not None:
                self.x_train, self.x_test, self.x_val = self.formatter.transform_inputs(self.x_train,
                                                    self.x_test,
                                                    self.x_val)
            else:
                self.x_train, self.x_test = self.formatter.transform_inputs(self.x_train,
                                                                            self.x_test)
        # specifying hidden layer size if there are more than 2 layers
        if len(self.hidden_size) != params["dnn_layers"]:
            for _ in range((params["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor(math.sqrt(self.num_features + self.num_class) + 5))
        params["hidden_size"] = self.hidden_size

        # tensorize the data
        x_train_nn = torch.Tensor(self.x_train).to(self.device)   
        x_test_nn = torch.Tensor(self.x_test).to(self.device)
        r_train_nn = torch.Tensor(self.r_train).to(self.device)
        r_test_nn = torch.Tensor(self.r_test).to(self.device)            

        # build the network
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

        # saving the solution that can be fed into mip model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "layer" in name:
                    self.beta_initial = param.data.detach().cpu().clone().numpy() 
    
        # saving the model
        torch.save(self.model.state_dict(), self.formatter.model_path+"_cslr.pt")
        
        if self.expt == "cslr":
            return (self.return_results(train_probs, test_probs, val_probs)
                    if self.x_val is not None
                    else (self.return_results(train_probs, test_probs)))
    
    def mip_opt(self):
        """Building linear programming models, mip or mip-wi, for EDCS problems
            
        Returns:
            The function that calculates the results.
        """
        
        # building the model, mip-wi or mip
        if self.expt == "mip_wi":
            # if no initial solution is present, call cslr
            if self.beta_initial is None:
                self.gradient()
            #updating the inputs in case of infeasibility
            self.opt_initial(self.beta_initial)
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
            for i in range(self.num_class):
                for j in range(self.num_features):
                    # providing the initial solution
                    self.model_mip.beta[i,j].start = self.beta_initial[i,j]
            self.model_mip.m.update() 
        else:
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
        
        path = self.formatter.model_path+"_"+self.expt
        self.mip_soln = {}
        self.model_mip.m.modelSense = GRB.MAXIMIZE
        self.model_mip.m.setParam(GRB.Param.TimeLimit, self.time_limit)
        self.model_mip.m.setParam(GRB.Param.LogFile, path+"_log")
        self.model_mip.m.write(path+'.lp')
        
        # optimizing until first feasible solution is found
        if self.analyze_first_feasible == True:
            if self.expt == "mip_wi":
                self.model_mip.m.setParam(GRB.Param.SolutionLimit, 2)
            else:
                self.model_mip.m.setParam(GRB.Param.SolutionLimit, 1)
            self.model_mip.m.optimize()
            mip_gap, obj, time_passed = (self.model_mip.m.MIPGap,
                                         self.model_mip.m.objVal,
                                         self.model_mip.m.RunTime)
            self.mip_soln["first_feasible"] = {
                "mip_gap": mip_gap,
                "obj_val": obj,
                "norm_obj_val": obj/sum(self.rmax_train),
                "time_passed": time_passed
            }
            
            # continuing optimization until the time limit
            if self.first_feasible_continue == True:
                runTime = self.model_mip.m.RunTime
                new_time_limit = self.time_limit - runTime
                
                # continue if there is still time left
                if new_time_limit > 0:
                    self.model_mip.m.setParam(GRB.Param.TimeLimit, new_time_limit)
                    self.model_mip.m.setParam(GRB.Param.SolutionLimit, 2000000000)
                    self.model_mip.m.optimize()
                    runTime = runTime + self.model_mip.m.RunTime
                    mip_gap, obj, time_passed = self.model_mip.m.MIPGap, self.model_mip.m.objVal, runTime
                    self.mip_soln["last_feasible"] = {
                        "mip_gap": mip_gap,
                        "obj_val": obj,
                        "norm_obj_val": obj/sum(self.rmax_train),
                        "time_passed": time_passed
                    }
            
                # ending the optimization if already passed the time limit
                else:
                    self.mip_soln["last_feasible"] = self.mip_soln["first_feasible"]

            # ending the optimization
            else:
                self.mip_soln["last_feasible"] = self.mip_soln["first_feasible"]
        
        # optimizing until the time limit
        else:
            self.model_mip.m.optimize()
            mip_gap, obj, time_passed = (self.model_mip.m.MIPGap,
                                         self.model_mip.m.objVal,
                                         self.model_mip.m.RunTime)
            self.mip_soln["last_feasible"] = {
                "mip_gap": mip_gap,
                "obj_val": obj,
                "norm_obj_val": obj/sum(self.rmax_train),
                "time_passed": time_passed
            }
            self.mip_soln["first_feasible"] = self.mip_soln["last_feasible"]
        
        # calculating the probabilities for each set and printing the coefficients
        test_probs = np.zeros((len(self.x_test), self.num_class))
        train_probs = np.zeros((len(self.x_train), self.num_class))
        if self.x_val is not None:
            val_probs = np.zeros((len(self.x_val), self.num_class))
        try:
            vars_beta = np.zeros((self.num_class, self.num_features))
            for k in range(self.num_class):
                for j in range(self.num_features):
                    vars_beta[k,j] = self.model_mip.beta[k,j].x
            pickle.dump(vars_beta, open(path+"_params.dat", "wb"))
            
            for i in range(len(self.x_test)):
                for k in range(self.num_class):
                    test_probs[i,k] = sum(self.x_test[i,j] * vars_beta[k,j]
                                          for j in range(self.num_features))
            for i in range(len(self.x_train)):
                for k in range(self.num_class):
                    train_probs[i,k] = sum(self.x_train[i,j] * vars_beta[k,j]
                                           for j in range(self.num_features))  
            if self.x_val is not None:
                for i in range(len(self.x_val)):
                    for k in range(self.num_class):
                        val_probs[i,k] = sum(self.x_val[i,j] * vars_beta[k,j]
                                             for j in range(self.num_features))
            return (self.return_results(train_probs, test_probs, val_probs)
                    if self.x_val is not None
                    else (self.return_results(train_probs, test_probs)))
        except:
            return (self.return_results(train_probs, test_probs, val_probs)
                    if self.x_val is not None
                    else (self.return_results(train_probs, test_probs)))
    
    def result(self):
        """Returns the results.
            
        Returns:
            Returns the results of the specified model.
        """
        
        if self.expt == "cslr":
            
            return self.gradient()
                
        if self.expt == "mip_wi" or self.expt == "mip":
                    
            return self.mip_opt()
            
    def return_results(self, train_probs, test_probs, val_probs=None):
        """Evaluates the model outputs.
            
        Args:
            train_probs: probabilities of the classes - train set
            test_probs: probabilities of the classes - test set
            val_probs: probabilities of the classes - validation set
            
        Returns:
            Triple of results for: (train, test, validation)
            Results consist of normalized return, accuracy and weighted F score.
        """
        
        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, train_outcome = test_learning(train_probs, self.r_train)
        test_return = test_return / sum(self.rmax_test)
        train_return = train_return / sum(self.rmax_train)

        test_acc, test_f1 = (accuracy_score(self.y_test, test_outcome),
                             f1_score(self.y_test, test_outcome, average="weighted")
                            )   
        train_acc, train_f1 = (accuracy_score(self.y_train, train_outcome),
                               f1_score(self.y_train, train_outcome, average="weighted")
                              )  
        self.test_scores, self.train_scores = ([test_return, test_acc, test_f1],
                                     [train_return, train_acc, train_f1]
                                    )
        if self.expt == "mip_wi" or self.expt == "mip":
            self.train_scores.append(self.mip_soln)
        
        if val_probs is not None:
            
            val_return, val_outcome = test_learning(val_probs, self.r_val)
            val_return = val_return / sum(self.rmax_val)
            val_acc, val_f1 = (accuracy_score(self.y_val, val_outcome),
                                 f1_score(self.y_val, val_outcome, average="weighted"))  
            self.val_scores = [val_return, val_acc, val_f1]
            
        return self.train_scores, self.test_scores, self.val_scores
        

