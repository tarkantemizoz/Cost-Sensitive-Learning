#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xgboost as xgb
import numpy as np

from bayes_opt.bayes_linearnet import cslr_bayes
from bayes_opt.utils import test_learning

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from bayes_opt.bayes_logistic import bayes_logistic
from sklearn.tree import DecisionTreeClassifier
from bayes_opt.bayes_tree import bayes_tree
from bayes_opt.bayes_xgboost import bayes_xgboost      
           
class ml_models: 

    def __init__(self,
                 formatter,
                 train,
                 test,
                 valid=None
                ):
        
        self.formatter = formatter
        self.bayes = self.formatter.bayes
        self.fixed_params = self.formatter.get_experiment_params()        
        self.scaler = self.fixed_params.get("scaler", False)        
        self.xgsteps = 20
        self.expt = "logistic"
        
        self.x_train = train[0]
        self.x_test = test[0] 
        self.r_train = train[1]
        self.r_test = test[1]
        self.rmax_train = sum(train[2])        
        self.rmax_test = sum(test[2]) 
        self.y_train = train[3]     
        self.y_test = test[3]        
        self.x_val = (valid[0] if valid is not None and valid != [] else None)
        self.r_val = (valid[1] if valid is not None and valid != [] else None)
        self.rmax_val = (valid[2] if valid is not None and valid != [] else None)   
        self.y_val = (valid[3] if valid is not None and valid != [] else None)             
        self.train_scores = []
        self.test_scores = []
        self.val_scores = []
        
    def tree(self):
        
        if self.bayes == True:
            
            bayes = bayes_tree(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_tree = bayes.bayes()
            (depth, min_samples, cp) = (best_params_tree.get("max_depth", ""),
                                        best_params_tree.get("min_samples_leaf", ""),
                                        best_params_tree.get("ccp_alpha", ""))
            dt = DecisionTreeClassifier(min_samples_leaf = min_samples,
                                        ccp_alpha = cp,
                                        max_depth = depth).fit(self.x_train,
                                                               self.y_train)  
            train_probs = dt.predict_proba(self.x_train) 
            test_probs = dt.predict_proba(self.x_test) 
            
            if self.x_val is not None:
                val_probs = dt.predict_proba(x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
        
        else:                 
            
            dt = DecisionTreeClassifier().fit(self.x_train, self.y_train)
            train_probs = dt.predict_proba(self.x_train) 
            test_probs = dt.predict_proba(self.x_test) 
            
            if self.x_val is not None:
                val_probs = dt.predict_proba(x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
    
    def logistic(self):
        
        if self.scaler == True:
            if self.x_val is not None:
                x_train, x_test, x_val = self.formatter.transform_inputs(self.x_train,
                                                                         self.x_test,
                                                                         self.x_val)
            else:
                x_train, x_test = self.formatter.transform_inputs(self.x_train,
                                                                  self.x_test)         
        else:
            x_train, x_test, x_val = self.x_train, self.x_test, self.x_val
                  
        if self.bayes == True:
            
            bayes = bayes_logistic(self.formatter , x_train, self.y_train, self.r_train)
            best_params_logistic = bayes.bayes()
            c, penalty = best_params_logistic.get("c", ""), best_params_logistic.get("penalty", "")
            clf = (LogisticRegression(C = c,
                                      penalty=penalty,
                                      solver='saga')
                   if penalty != "elasticnet"
                   else LogisticRegression(C = c,
                                           penalty=penalty,
                                           solver='saga',
                                           l1_ratio = best_params_logistic.get("l1", ""))
                  )          
            clf_fit = clf.fit(x_train, self.y_train)        
            train_probs = clf_fit.predict_proba(x_train) 
            test_probs = clf_fit.predict_proba(x_test)  
            
            if self.x_val is not None:
                val_probs = clf_fit.predict_proba(x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
            
        else:            
            
            clf_fit = LogisticRegression().fit(x_train, self.y_train)        
            train_probs = clf_fit.predict_proba(x_train) 
            test_probs = clf_fit.predict_proba(x_test) 
            
            if self.x_val is not None:
                val_probs = clf_fit.predict_proba(x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)

    def xgboost(self):

        if self.bayes == True:
            bayes = bayes_xgboost(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_xgb = bayes.bayes()            
            param = {'eta' : best_params_xgb.get("eta", ""), 
                     'max_depth' : best_params_xgb.get("max_depth", ""),
                     'min_child_weight' : best_params_xgb.get("min_child_weight", ""),
                     'gamma' : best_params_xgb.get("gamma", ""),
                     'colsample_bytree' : best_params_xgb.get("colsample_bytree", ""),
                     'objective': 'multi:softprob',
                     'num_class': self.r_train.shape[1]
                    }                 
            model = xgb.train(param, xgb.DMatrix(self.x_train,
                                                 label=self.y_train),
                              self.xgsteps)                      
            train_probs = model.predict(xgb.DMatrix(self.x_train, label=self.y_train))
            test_probs = model.predict(xgb.DMatrix(self.x_test, label=self.y_test))
            
            if self.x_val is not None:
                val_probs = model.predict(xgb.DMatrix(self.x_val, label=self.y_val)) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs) 
            
        else:
            
            dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
            dtest = xgb.DMatrix(self.x_test, label=self.y_test)  
            param = {'objective': 'multi:softprob', 'num_class': self.r_train.shape[1]}
            train_probs = xgb.train(param, dtrain, self.xgsteps).predict(dtrain)
            test_probs = xgb.train(param, dtrain, self.xgsteps).predict(dtest)
                        
            if self.x_val is not None:
                dval = xgb.DMatrix(self.x_val, label=self.y_val)
                val_probs = xgb.train(param, dtrain, self.xgsteps).predict(dval)
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)   
           

    def result(self):
                
        if self.expt == "logistic":
            
            return self.logistic()
        
        if self.expt == "tree":
            
            return self.tree()
        
        if self.expt == "xgboost":
            
            return self.xgboost()        
        
     
    def return_results(self, train_probs, test_probs, val_probs=None):
        
        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, train_outcome = test_learning(train_probs, self.r_train)
        test_return = test_return / self.rmax_test
        train_return = train_return / self.rmax_train 
        test_acc, test_f1 = (accuracy_score(self.y_test, test_outcome),
                             f1_score(self.y_test, test_outcome, average="weighted")
                            )   
        train_acc, train_f1 = (accuracy_score(self.y_train, train_outcome),
                               f1_score(self.y_train, train_outcome, average="weighted")
                              )  
        self.test_scores, self.train_scores = ([test_return, test_acc, test_f1],
                                     [train_return, train_acc, train_f1]
                                    )
        
        if val_probs is not None:
            
            val_return, val_outcome = test_learning(val_probs, self.r_val)
            val_return = val_return / self.rmax_val
            val_acc, val_f1 = (accuracy_score(self.y_val, val_outcome),
                                 f1_score(self.y_val, val_outcome, average="weighted"))  
            self.val_scores = [val_return, val_acc, val_f1]
            
        return self.train_scores, self.test_scores, self.val_scores
        
     
        

