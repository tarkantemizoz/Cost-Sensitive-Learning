#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import optuna
import numpy as np

from sklearn.linear_model import LogisticRegression
from bayes_opt.utils import give_set, give_score, test_learning

class bayes_logistic: 

    def __init__(self, config, data, labels, returns):    
    
        self.cval = config.get("cval", False)
        self.n_val_splits = config.get("n_val_splits", 10)    
        self.rep = config.get("rep", 0) 
        self.set_num = config.get("set", 0) 
        self.n_trials = config.get("n_trials", 2)
        self.which_score = config.get("which_score", "gain")
        self.val_sets = config.get("val_sets", "[]")
        self.data = data
        self.labels = labels  
        self.returns = returns    

    def train_bayes_logistic(self, trial):

        space = {'c': trial.suggest_loguniform('c', 0.01, 100),
               'penalty': trial.suggest_categorical('penalty',["l1", "l2", "elasticnet"]),
        }
        if space['penalty'] == "elasticnet":      
            space['l1'] = trial.suggest_uniform('l1', 0, 1)
        
        if self.cval == False:
            if space["penalty"] != "elasticnet":
                clf = LogisticRegression(C = space['c'], penalty = space["penalty"],
                                         solver='saga', max_iter=500).fit(data, labels)
            else:
                clf = LogisticRegression(C = space['c'], penalty = space["penalty"],
                                         solver='saga',max_iter=500, l1_ratio = space['l1']).fit(data,labels)
                                                                                                 
            probs = clf.predict_proba(self.data)
            test_return, test_outcome = test_learning(probs, self.returns)
            score = give_score(test_return, test_outcome, labels, self.which_score)
        
        else:
            
            score = np.zeros(self.n_val_splits)
            count = 0

            for i in range(self.n_val_splits):
                
                train_index, test_index = give_set(self.rep, self.set_num, i, self.val_sets)
                X_train, X_test = self.data[train_index], self.data[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                r_train, r_test = self.returns[train_index], self.returns[test_index]
                if space["penalty"] != "elasticnet":
                    clf = LogisticRegression(C = space['c'], penalty = space["penalty"],
                                             solver='saga').fit(X_train, y_train)
                else:
                    clf = LogisticRegression(C = space['c'], penalty = space["penalty"],
                                             solver='saga', l1_ratio = space['l1']).fit(X_train, y_train)
                probs = clf.predict_proba(X_test)
                test_return, test_outcome = test_learning(probs, r_test)
                score[count] = give_score(test_return, test_outcome, y_test, self.which_score)
                count += 1

        return np.mean(score)  

    def bayes(self):

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_logistic, n_trials=self.n_trials)
    
        return study.best_params

