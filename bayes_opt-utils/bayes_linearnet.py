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


import optuna
import torch
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from bayes_opt.utils import test_learning

class cslr_bayes: 
    """ Hyperparameter optimisation for CSLR using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
        device: device to store the data, cpu or cuda
        scaler: whether to scale the data
        batch_norm: whether to apply batch normalization on the network
        n_epochs: total number of epochs.
        n_steps: number of epochs to evaluate the results
        batch_size: batch-size for the network
        dnn_layers: number of layers of the network
        hidden_size: size of the hidden layers of the network
        bayes_params: bayesian optimization parameters
        bayes_trials: number of trials
        inner_cval: whether to apply inner-cross validation
        n_val_splits: number of inner-cross validation folds
        batch_size_bayes: set of batch-size to be optimized
        dnn_layers_bayes: set of number of layers to be optimized
    """

    def __init__(self, formatter, data, labels, returns, config):
        """Instantiates the attributes and parameters.
            
        Args:
            formatter: formatter for the specified experiment
            data: the data set
            labels: labels of the data
            returns: returns of the data
            config: configuration for the experiment
        """
        
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
        self.labels = labels
        self._max_batch = math.floor(math.log2(self.data.shape[0]))
    
    def train_bayes_linear(self, trial):                 
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'lr_rate' : (trial.suggest_uniform('lr_rate', 0.00005, 0.01)),
                'batch_size': (trial.suggest_int('batch_size',
                                                 self.batch_size_bayes[0],
                                                 self.batch_size_bayes[1])
                               if self.batch_size_bayes is not None
                               else self._max_batch),
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
        
        # specifying hidden layer size if there are more than 2 layers
        if len(self.hidden_size) != config_nn["dnn_layers"]:
            for _ in range((config_nn["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor
                                        (math.sqrt(config_nn["n_inputs"]
                                                   + config_nn["n_outputs"]) + 5
                                        )
                                       )
        config_nn["hidden_size"] = self.hidden_size
        
        # maximize the train results, no inner-cross validation
        if self.inner_cval == False:
            
            x = self.data
            # calling the prespecified scaler, which depends on the experiment
            if self.scaler == True:
                x = self.formatter.transform_inputs(x)
            
            # tensorize the data
            x_nn = torch.Tensor(x).to(self.device)   
            r_nn = torch.Tensor(self.returns).to(self.device)
            
            # build the network
            model = LinearNet(config_nn).to(self.device)
            config_nn["batch_size"] = (2**space['batch_size']
                                       if space['batch_size']<self._max_batch
                                       else len(x))                     
            optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
            optimization = Optimization(model, optimizer, config_nn)
            optimization.train(x_nn, r_nn)
            _, _, test_probs = optimization.evaluate(x_nn, r_nn)
            result,_ = test_learning(test_probs.detach().cpu().clone().numpy(),
                                        self.returns)

        # apply inner-cross validation
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)

            for train_index, test_index in skf.split(self.data, self.labels):
                
                x_train, x_test = self.data[train_index], self.data[test_index]
                r_train, r_test = self.returns[train_index], self.returns[test_index]
                
                # calling the prespecified scaler, which depends on the experiment
                if self.scaler == True:
                    x_train, x_test = self.formatter.transform_inputs(x_train, x_test)
                
                # tensorize the data
                x_train = torch.Tensor(x_train).to(self.device)   
                x_test = torch.Tensor(x_test).to(self.device)
                r_train = torch.Tensor(r_train).to(self.device)
                r_test = torch.Tensor(r_test).to(self.device)                 
                
                # build the network
                model = LinearNet(config_nn).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=space['lr_rate'])
                self._max_batch = math.floor(math.log2(x_train.shape[0]))
                config_nn["batch_size"] = (2**space['batch_size']
                                           if space['batch_size']<self._max_batch
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
        """Building bayesian optimization environment.
            
        Returns:
            the optimal hyperparameters thus far
        """

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_linear, n_trials=self.bayes_trials)
    
        return study.best_params

