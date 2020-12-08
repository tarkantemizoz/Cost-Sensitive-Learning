#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
import numpy as np
import xgboost as xgb

from bayes_opt.utils import test_learning
from sklearn.model_selection import KFold

class bayes_xgboost: 

    def __init__(self, formatter, data, labels, returns):    
    
        self.formatter = formatter      
        self.bayes_params = self.formatter.get_bayes_params()    
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.inner_cval = self.bayes_params["inner_cval"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10) 
        self.xgsteps = 20
        
        self.data = data
        self.labels = labels  
        self.returns = returns        
    
    def train_bayes_boost(self, trial):

        space = {'eta': trial.suggest_uniform('eta', 0.01, 0.4),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_uniform('gamma', 0, 0.4),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
                'steps': self.xgsteps
                }

        param = {'eta' : space['eta'], 
                'max_depth' : space['max_depth'],
                'min_child_weight' : space['min_child_weight'],
                'gamma' : space['gamma'],
                'colsample_bytree' : space['colsample_bytree'],
                'objective': 'multi:softprob',
                'num_class': self.returns.shape[1]
                }        
        
        if self.inner_cval == False:
            
            D_train = xgb.DMatrix(self.data, label=self.labels)
            model = xgb.train(param, D_train, space['steps'])
            probs = model.predict(D_train)
            result,_ = test_learning(probs, self.returns)
        
        else:
            
            test_return = np.zeros(self.n_val_splits)
            count = 0
            skf = KFold(self.n_val_splits, shuffle=True)
            
            for train_index, test_index in skf.split(self.data):
                
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

        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=self.train_bayes_boost, n_trials=self.bayes_trials)
    
        return study.best_params

