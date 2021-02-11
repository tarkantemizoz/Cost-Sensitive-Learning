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
from sklearn.tree import DecisionTreeClassifier
from bayes_opt.utils import test_learning

class bayes_tree: 
    """ Hyperparameter optimisation for Decision Trees using Bayesian Optimization.
        
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

        self.data = data
        self.labels = labels  
        self.returns = returns    
        
    def train_bayes_tree(self, trial):
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
                'ccp_alpha': trial.suggest_uniform('ccp_alpha', 0.00001, 0.01)
                }
        
        # maximize the train results, no inner-cross validation
        if self.inner_cval == False:
        
            dt = DecisionTreeClassifier(min_samples_leaf = space['min_samples_leaf'],
                                        ccp_alpha = space['ccp_alpha'],
                                        max_depth = space['max_depth']).fit(self.data, self.labels)  
            probs = dt.predict_proba(self.data)
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
                
                dt = DecisionTreeClassifier(min_samples_leaf = space['min_samples_leaf'],
                                            ccp_alpha = space['ccp_alpha'],
                                            max_depth = space['max_depth']).fit(x_train, y_train) 
                probs = dt.predict_proba(x_test)
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
        study.optimize(func=self.train_bayes_tree, n_trials=self.bayes_trials)
    
        return study.best_params

