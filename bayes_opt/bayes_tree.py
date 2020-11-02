#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import optuna
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from bayes_opt.utils import give_set, give_score, test_learning

class bayes_tree: 

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
        
    def train_bayes_tree(self, trial):

        space = {'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
                'ccp_alpha': trial.suggest_uniform('ccp_alpha', 0.00001, 0.01)
                }
        
        if self.cval == False:
        
            dt = DecisionTreeClassifier(min_samples_leaf = space['min_samples_leaf'],ccp_alpha = space['ccp_alpha'],
                                        max_depth = space['max_depth']).fit(self.data, self.labels)  
            probs = dt.predict_proba(self.data)
            test_return, test_outcome = test_learning(probs, self.returns)
            score = give_score(test_return, test_outcome, self.labels, self.which_score)
        
        else:
            
            score = np.zeros(self.n_val_splits)
            count = 0

            for i in range(self.n_val_splits):
                
                train_index, test_index = give_set(self.rep, self.set_num, i, self.val_sets)
                X_train, X_test = self.data[train_index], self.data[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                r_train, r_test = self.returns[train_index], self.returns[test_index]
                
                dt = DecisionTreeClassifier(min_samples_leaf = space['min_samples_leaf'],ccp_alpha = space['ccp_alpha'],
                                            max_depth = space['max_depth']).fit(X_train, y_train) 
                probs = dt.predict_proba(X_test)
                test_return, test_outcome = test_learning(probs, r_test)
                score[count] = give_score(test_return, test_outcome, y_test, self.which_score)
                count += 1

        return np.mean(score)  

    def bayes(self):

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_tree, n_trials=self.n_trials)
    
        return study.best_params

