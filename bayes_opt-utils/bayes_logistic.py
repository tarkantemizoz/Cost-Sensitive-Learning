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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from bayes_opt.utils import test_learning

class bayes_logistic: 
    """ Hyperparameter optimisation for Logistic Regression using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
        scaler: whether to scale the data
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
        self.scaler = self.formatter.scaler

        self.data = data
        self.labels = labels  
        self.returns = returns    

    def train_bayes_logistic(self, trial):
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'c': trial.suggest_loguniform('c', 0.01, 100),
               'penalty': trial.suggest_categorical('penalty', ["l1", "l2", "elasticnet"]),
        }
        
        if space['penalty'] == "elasticnet":      
            space['l1'] = trial.suggest_uniform('l1', 0, 1)
        
        # maximize the train results, no inner-cross validation
        if self.inner_cval == False:

            x = self.data
            # calling the prespecified scaler, which depends on the experiment
            if self.scaler == True:
                x = self.formatter.transform_inputs(x)   
                
            if space["penalty"] != "elasticnet":
                clf = LogisticRegression(C = space['c'],
                                         penalty = space["penalty"],
                                         solver='saga',
                                         max_iter=500).fit(x, self.labels)
            else:
                clf = LogisticRegression(C = space['c'],
                                         penalty = space["penalty"],
                                         solver='saga',
                                         max_iter=500,
                                         l1_ratio = space['l1']).fit(x, self.labels)
                                                                                                 
            probs = clf.predict_proba(x)
            result,_ = test_learning(probs, self.returns)

        # apply inner-cross validation
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)
            
            for train_index, test_index in skf.split(self.data, self.labels):
                
                x_train, x_test = self.data[train_index], self.data[test_index]
                _, r_test = self.returns[train_index], self.returns[test_index]
                y_train, _ = self.labels[train_index], self.labels[test_index]

                x_tr, x_te = x_train, x_test
                # calling the prespecified scaler, which depends on the experiment
                if self.scaler == True:
                    x_tr, x_te = self.formatter.transform_inputs(x_tr, x_te)  
                
                if space["penalty"] != "elasticnet":
                    clf = LogisticRegression(C = space['c'],
                                             penalty = space["penalty"],
                                             solver='saga').fit(x_tr, y_train)
                else:
                    clf = LogisticRegression(C = space['c'],
                                             penalty = space["penalty"],
                                             solver='saga',
                                             l1_ratio = space['l1']).fit(x_tr, y_train)
                    
                probs = clf.predict_proba(x_te)
                test_return[count],_ = test_learning(probs, r_test)
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
        study.optimize(func=self.train_bayes_logistic, n_trials=self.bayes_trials)
    
        return study.best_params

