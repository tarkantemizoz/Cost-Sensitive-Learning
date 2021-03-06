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
    """Machine learning algorithms for EDCS problems
        
    Attributes:
        formatter: formatter for the specified experiment
        expt: name of ml method
        train_scores: a triple of: (normalized return, accuracy, weighted F score)
            for train data
        test_scores: a triple of: (normalized return, accuracy, weighted F score)
            for test data
        val_scores: a triple of: (normalized return, accuracy, weighted F score)
            for validation data
        x_train: features for train set
        r_train: returns for train set
        rmax_train: maximum returns for train set
        y_train: labels for train set
        Same applies for test and validation sets.
    """
    
    def __init__(self,
                 formatter,
                 train,
                 test,
                 valid=None
                ):
        """Organizes the data for machine learning algorithms.
            
        Args:
            formatter: formatter for the specified experiment
            train: holds the quadruple of: (features, returns, maximum returns, labels)
                for train data
            test: holds the quadruple of: (features, returns, maximum returns, labels)
                for test data
            valid: holds the quadruple of: (features, returns, maximum returns, labels)
                for validation data
        """
        
        self.formatter = formatter
        self._xgsteps = 20
        self.expt = "logistic"
        
        self.x_train = train[0]
        self.x_test = test[0] 
        self.r_train = train[1]
        self.r_test = test[1]
        self.rmax_train = train[2]       
        self.rmax_test = test[2] 
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
        """Runs decision trees on the input.
            
        Returns:
            The function that calculates the results.
        """
        
        # applying bayesian optimization
        if self.formatter.bayes == True:
            
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
                val_probs = dt.predict_proba(self.x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
    
        # calling the default model
        else:                 

            dt = DecisionTreeClassifier(max_depth = 8).fit(self.x_train, self.y_train)
            train_probs = dt.predict_proba(self.x_train) 
            test_probs = dt.predict_proba(self.x_test) 
            
            if self.x_val is not None:
                val_probs = dt.predict_proba(self.x_val) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
    
    def logistic(self):
        """Runs logistic regression on the input.
            
        Returns:
            The function that calculates the results.
        """
        
        # calling the prespecified scaler, which depends on the experiment
        if self.formatter.get_experiment_params().get("scaler", False) == True:
            if self.x_val is not None:
                x_train, x_test, x_val = self.formatter.transform_inputs(self.x_train,
                                                                         self.x_test,
                                                                         self.x_val)
            else:
                x_train, x_test = self.formatter.transform_inputs(self.x_train,
                                                                  self.x_test)         
        else:
            x_train, x_test, x_val = self.x_train, self.x_test, self.x_val

        # applying bayesian optimization
        if self.formatter.bayes == True:
            
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

                    # calling the default model
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
        """Runs xgboost on the input.
            
        Returns:
           The function that calculates the results.
        """
        # applying bayesian optimization
        if self.formatter.bayes == True:
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
                              self._xgsteps
                              )
            train_probs = model.predict(xgb.DMatrix(self.x_train, label=self.y_train))
            test_probs = model.predict(xgb.DMatrix(self.x_test, label=self.y_test))
            
            if self.x_val is not None:
                val_probs = model.predict(xgb.DMatrix(self.x_val, label=self.y_val)) 
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)
    
        # calling the default model
        else:

            dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
            dtest = xgb.DMatrix(self.x_test, label=self.y_test)  
            param = {'objective': 'multi:softprob', 'num_class': self.r_train.shape[1],
                     'max_depth': 8
                    }
            model = xgb.train(param, dtrain, self._xgsteps)
            train_probs = model.predict(dtrain)
            test_probs = model.predict(dtest)
                        
            if self.x_val is not None:
                dval = xgb.DMatrix(self.x_val, label=self.y_val)
                val_probs = model.predict(dval)
                return self.return_results(train_probs, test_probs, val_probs)
            else:
                return self.return_results(train_probs, test_probs)   
           

    def result(self):
        """Returns the results.
            
        Returns:
            Returns the results of the specified model.
        """
        
        if self.expt == "logistic":
            
            return self.logistic()
        
        if self.expt == "tree":
            
            return self.tree()
        
        if self.expt == "xgboost":
            
            return self.xgboost()        
        
     
    def return_results(self, train_probs, test_probs, val_probs=None):
        """Evaluates the model outputs.
            
        Args:
            train_probs: probabilities of the classes - train set
            test_probs: probabilities of the classes - test set
            val_probs: probabilities of the classes - validation set
            
        Returns:
            Triple of results for: (train, test, validation)
            Results consist of normalized return, accuracy and weighted F score.
        """

        test_return, test_outcome = test_learning(test_probs, self.r_test)
        train_return, train_outcome = test_learning(train_probs, self.r_train)
        test_return = test_return / sum(self.rmax_test)
        train_return = train_return / sum(self.rmax_train) 
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
            val_return = val_return / sum(self.rmax_val)
            val_acc, val_f1 = (accuracy_score(self.y_val, val_outcome),
                                 f1_score(self.y_val, val_outcome, average="weighted"))  
            self.val_scores = [val_return, val_acc, val_f1]
            
        return self.train_scores, self.test_scores, self.val_scores
        
     
        

