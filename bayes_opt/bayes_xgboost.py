#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
import numpy as np
import xgboost as xgb

from bayes_opt.utils import give_set, give_score, test_learning

class bayes_xgboost: 

    def __init__(self, config, data, labels, returns):    
    
        self.cval = config.get("cval", False)
        self.n_val_splits = config.get("n_val_splits", 10)    
        self.rep = config.get("rep", 0) 
        self.set_num = config.get("set", 0) 
        self.n_trials = config.get("n_trials", 2)
        self.which_score = config.get("which_score", "gain")
        self.val_sets = config.get("val_sets", "[]")
        self.n_steps = config.get("n_steps", 20)
        self.data = data
        self.labels = labels  
        self.returns = returns    
    
    def train_bayes_boost(self, trial):

        space = {'eta': trial.suggest_uniform('eta', 0.01, 0.4),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_uniform('gamma', 0, 0.4),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
                'steps': self.n_steps
                }

        param = {'eta' : space['eta'], 
                'max_depth' : space['max_depth'],
                'min_child_weight' : space['min_child_weight'],
                'gamma' : space['gamma'],
                'colsample_bytree' : space['colsample_bytree'],
                'objective': 'multi:softprob',
                'num_class': self.returns.shape[1]}        
        
        if self.cval == False:
            
            D_train = xgb.DMatrix(self.data, label=self.labels)
            model = xgb.train(param, D_train, space['steps'])
            probs = model.predict(D_train)
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
                D_train = xgb.DMatrix(X_train, label=y_train)
                D_test = xgb.DMatrix(X_test, label=y_test)          
                model = xgb.train(param, D_train, space['steps'])
                probs = model.predict(D_test)
                test_return, test_outcome = test_learning(probs, r_test)
                score[count] = give_score(test_return, test_outcome, y_test, self.which_score)   
                count += 1

        return np.mean(score)  

    def bayes(self):

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_boost, n_trials=self.n_trials)
    
        return study.best_params

