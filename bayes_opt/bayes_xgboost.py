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
import numpy as np
import xgboost as xgb

from bayes_opt.utils import test_learning
from sklearn.model_selection import StratifiedKFold

class bayes_xgboost: 
    """ Hyperparameter optimisation for Logistic Regression using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
        bayes_params: bayesian optimization parameters
        bayes_trials: number of trials
        inner_cval: whether to apply inner-cross validation
        n_val_splits: number of inner-cross validation folds
    """
    
    def __init__(self, formatter, data, labels, returns):    
        """Instantiates the attributes and parameters.
            
        Args:
            formatter: formatter for the specified experiment
            data: the data set
            labels: labels of the data
            returns: returns of the data
        """
        
        self.formatter = formatter      
        self.bayes_params = self.formatter.get_bayes_params()    
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.inner_cval = self.bayes_params["inner_cval"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10) 
        self._xgsteps = 20
        
        self.data = data
        self.labels = labels  
        self.returns = returns        
    
    def train_bayes_boost(self, trial):
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'eta': trial.suggest_uniform('eta', 0.01, 0.4),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_uniform('gamma', 0, 0.4),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
                'steps': self._xgsteps
                }

        param = {'eta' : space['eta'], 
                'max_depth' : space['max_depth'],
                'min_child_weight' : space['min_child_weight'],
                'gamma' : space['gamma'],
                'colsample_bytree' : space['colsample_bytree'],
                'objective': 'multi:softprob',
                'num_class': self.returns.shape[1]
                }
        
        # maximize the train results, no inner-cross validation
        if self.inner_cval == False:
            
            D_train = xgb.DMatrix(self.data, label=self.labels)
            model = xgb.train(param, D_train, space['steps'])
            probs = model.predict(D_train)
            result,_ = test_learning(probs, self.returns)
        
        # apply inner-cross validation
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)

            for train_index, test_index in skf.split(self.data, self.labels):
                
                x_train, x_test = self.data[train_index], self.data[test_index]
                _, r_test = self.returns[train_index], self.returns[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                
                D_train = xgb.DMatrix(x_train, label=y_train)
                D_test = xgb.DMatrix(x_test, label=y_test)          
                model = xgb.train(param, D_train, space['steps'])
                probs = model.predict(D_test)
                test_return[count],_ = test_learning(probs, r_test)
                count += 1

            result = np.mean(test_return)
                
        return result

    def bayes(self):
        """Building bayesian optimization environment.
            
        # apply inner-cross validationReturns:
            the optimal hyperparameters thus far
        """
        
        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_boost, n_trials=self.bayes_trials)
    
        return study.best_params

