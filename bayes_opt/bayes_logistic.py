#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import optuna
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from bayes_opt.utils import test_learning

class bayes_logistic: 

    def __init__(self, formatter, data, labels, returns):    
    
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

        space = {'c': trial.suggest_loguniform('c', 0.01, 100),
               'penalty': trial.suggest_categorical('penalty', ["l1", "l2", "elasticnet"]),
        }
        
        if space['penalty'] == "elasticnet":      
            space['l1'] = trial.suggest_uniform('l1', 0, 1)
        
        if self.inner_cval == False:

            x = self.data
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
        
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = KFold(self.n_val_splits, shuffle=True)
            
            for train_index, test_index in skf.split(self.data):
                
                x_train, x_test = self.data[train_index], self.data[test_index]
                _, r_test = self.returns[train_index], self.returns[test_index]
                y_train, _ = self.labels[train_index], self.labels[test_index]

                x_tr, x_te = x_train, x_test
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

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_logistic, n_trials=self.bayes_trials)
    
        return study.best_params

