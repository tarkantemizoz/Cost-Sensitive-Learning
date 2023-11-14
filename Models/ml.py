import xgboost as xgb
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from utils.bayes_logistic import bayes_logistic
from utils.bayes_svm import bayes_svm
from utils.bayes_svm_cost import bayes_svm_cost
from utils.bayes_tree import bayes_tree
from utils.bayes_xgboost import bayes_xgboost    
from utils.utils import test_performance
from Models.svm_cost import SVM
from Models.cslr import cslr

import costcla

from costcla.models import CostSensitiveRandomForestClassifier
from costcla.models import CostSensitiveRandomPatchesClassifier
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.models import CostSensitiveLogisticRegression    
from costcla.models import CostSensitivePastingClassifier
from costcla.models import CostSensitiveBaggingClassifier


class ml_models: 
    """Machine learning algorithms for EDCS problems
        
    Attributes:
        formatter: formatter for the specified experiment
        expt: name of ml method
        x_train: features for train set
        r_train: returns for train set
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
            train: holds the tuple of: (features, returns, labels)
                for train data
            test: holds the tuple of: (features, returns, labels)
                for test data
            valid: holds the tuple of: (features, returns, labels)
                for validation data
        """
        
        self.formatter = formatter
        self.expt = "logistic"
        self.cslr_model = cslr(formatter, train, test, valid)
        
        self.x_train, self.x_test = train[0], test[0]
        self.r_train, self.r_test = train[1], test[1]
        self.y_train, self.y_test = train[3], test[3]     
        self.x_val = (valid[0] if valid is not None and valid != [] else None)
        self.r_val = (valid[1] if valid is not None and valid != [] else None)
        self.y_val = (valid[3] if valid is not None and valid != [] else None)             

        self.svm_model = None
        self.logistic_model = None
        self.svm_cost_model = None

    def tree(self):
        """Runs decision trees on the input.
            
        Returns:
            The function that calculates the results.
        """

        # applying bayesian optimization
        if self.formatter.bayes == True:
            bayes = bayes_tree(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_tree = bayes.bayes()
            (depth, min_samples, cp, crit) = (best_params_tree.get("max_depth", ""),
                                                   best_params_tree.get("min_samples_leaf", ""),
                                                   best_params_tree.get("ccp_alpha", ""),
                                                   best_params_tree.get("criterion", ""))
            dt = DecisionTreeClassifier(min_samples_leaf = min_samples,
                                        ccp_alpha = cp,
                                        max_depth = depth,
                                        criterion = crit).fit(self.x_train, self.y_train)  
    
        # calling the default model
        else:                 
            dt = DecisionTreeClassifier().fit(self.x_train, self.y_train)
        
        return self.return_results(dt, self.x_train, self.x_test, self.x_val)
     
    def logistic(self, return_results = True):
        """Runs logistic regression on the input.
            
        Returns:
            The function that calculates the results.
        """
        
        # calling the prespecified scaler, which depends on the experiment
        x_train, x_test, x_val = self.format_data()

        # applying bayesian optimization
        if self.formatter.bayes == True:
            
            bayes = bayes_logistic(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_logistic = bayes.bayes()
            c, penalty = best_params_logistic.get("c", ""), best_params_logistic.get("penalty", "")
            clf = (LogisticRegression(C = c, penalty=penalty, solver='saga')
                   if penalty != "elasticnet"
                   else LogisticRegression(C = c, penalty="elasticnet", solver='saga', l1_ratio = best_params_logistic.get("l1", ""))
                  )          
            clf_fit = clf.fit(x_train, self.y_train)        

        # calling the default model
        else:    
            clf_fit = LogisticRegression().fit(x_train, self.y_train)  

        self.logistic_model = clf_fit

        if (return_results):
            return self.return_results(clf_fit, x_train, x_test, x_val)

    def xgboost(self):
        """Runs xgboost on the input.
            
        Returns:
           The function that calculates the results.
        """
        
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test)  
        dval = xgb.DMatrix(self.x_val, label=self.y_val)
        
        param = {}
        if self.r_train.shape[1] == 2:
            param["objective"] = "binary:hinge"
        else:
            param["objective"] = "multi:softmax"
            param["num_class"] = self.r_train.shape[1]
        param["early_stopping_rounds"] = 20
        param["verbosity"] = 0

        # applying bayesian optimization
        if self.formatter.bayes == True:
            bayes = bayes_xgboost(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_xgb = bayes.bayes()   

            param = {
                "subsample": best_params_xgb.get("subsample", ""),
                "max_depth": best_params_xgb.get("max_depth", ""),
                "eta": best_params_xgb.get("eta", ""),
                "min_child_weight": best_params_xgb.get("min_child_weight", ""),
                "gamma": best_params_xgb.get("gamma", ""),
            }
            num_rounds = 100 * (0.4 / param["eta"])

            model = xgb.train(param, dtrain, num_boost_round=round(num_rounds))

        # calling the default model
        else:
            model = xgb.train(param, dtrain)

        return self.return_results(model, dtrain, dtest, dval)
           
    def supportvector(self, return_results = True):
        """Runs SVM on the input.
            
        Returns:
           The function that calculates the results.
        """

        # calling the prespecified scaler, which depends on the experiment
        x_train, x_test, x_val = self.format_data()

        # applying bayesian optimization
        if self.formatter.bayes == True:
            bayes = bayes_svm(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_svm = bayes.bayes()
            C = best_params_svm.get("C", "")
            svm_fit = svm.SVC(kernel = 'linear', C = 2 ** C).fit(x_train, self.y_train)
    
        # calling the default model
        else:
            svm_fit = svm.SVC(kernel = 'linear', C = 10).fit(x_train, self.y_train)
            
        self.svm_model = svm_fit

        if (return_results):
            return self.return_results(svm_fit, x_train, x_test, x_val)

    def svm_costsensitive(self, return_results = True):
        """Runs Cost Sensitive SVM on the input.
            
        Returns:
           The function that calculates the results.
        """
        
        # calling the prespecified scaler, which depends on the experiment 
        x_train, x_test, x_val = self.format_data()

        # applying bayesian optimization
        if self.formatter.bayes == True:
            bayes = bayes_svm(self.formatter , self.x_train, self.y_train, self.r_train)
            best_params_svm = bayes.bayes()
            C = best_params_svm.get("C", "")
            svm_model = SVM("linear", C = 2 ** C)
            svm_model.fit(x_train, self.y_train, self.r_train)
    
        # calling the default model
        else:
            svm_model = SVM("linear", C=10)
            svm_model.fit(x_train, self.y_train, self.r_train)
          
        self.svm_cost_model = svm_model
        
        if (return_results):
            return self.return_results(svm_model, x_train, x_test, x_val)
                
    def return_results(self, model, x_train, x_test, x_val=None):
        """Evaluates the model outputs.

        Returns:
            Triple of results for: (train, test, validation)
            Results consist of normalized return, savings, accuracy and weighted F score.
        """
        
        train_preds = model.predict(x_train).astype(int)
        test_preds = model.predict(x_test).astype(int)
        
        train_scores = test_performance(self.y_train, train_preds, self.r_train)
        test_scores = test_performance(self.y_test, test_preds, self.r_test)
        val_scores = []
        
        if self.x_val is not None:           
            val_preds = model.predict(x_val) 
          
            val_scores = test_performance(self.y_val, val_preds, self.r_val)
            
        return train_scores, test_scores, val_scores
          
    def format_data(self):

        if self.formatter.get_experiment_params().get("scaler", False) == True:
            if self.x_val is not None:
                x_train, x_test, x_val = self.formatter.transform_inputs(self.x_train, self.x_test, self.x_val)
            else:
                x_train, x_test = self.formatter.transform_inputs(self.x_train, self.x_test)  
                x_val = self.x_val
            return x_train, x_test, x_val
        else:
            return self.x_train, self.x_test, self.x_val
   
    def cost_cla(self):
        
        if len(self.r_train[0]) > 2:
            raise ValueError("Number of classes should not be larger than 2 when applying CostCla")
        else:
            self.cost_mat = np.zeros((len(self.r_train),4))
            for i in range(len(self.r_train)):
                if self.y_train[i] == 1:

                    self.cost_mat[i,0] = -self.r_train[i,1]
                    self.cost_mat[i,1] = -self.r_train[i,0]

                else:

                    self.cost_mat[i,0] = -self.r_train[i,1]
                    self.cost_mat[i,1] = -self.r_train[i,0]       

    def tree_costcla(self):

        self.cost_cla()                     
        dt = CostSensitiveDecisionTreeClassifier().fit(self.x_train, self.y_train, self.cost_mat)
        
        return self.return_results(dt, self.x_train, self.x_test, self.x_val)

    def rf_costcla(self):

        self.cost_cla()                       
        rf = CostSensitiveRandomForestClassifier().fit(self.x_train, self.y_train, self.cost_mat)
        
        return self.return_results(rf, self.x_train, self.x_test, self.x_val) 
    
    def rp_costcla(self):

        self.cost_cla()                         
        rp = CostSensitiveRandomPatchesClassifier().fit(self.x_train, self.y_train, self.cost_mat)
        
        return self.return_results(rp, self.x_train, self.x_test, self.x_val) 
    
    def logistic_costcla(self):

        x_train, x_test, x_val = self.format_data()
        self.cost_cla()             
        logit = CostSensitiveLogisticRegression().fit(x_train, self.y_train, self.cost_mat)
        
        return self.return_results(logit, x_train, x_test, x_val) 
 
    def pasting_costcla(self):

        self.cost_cla()             
        pst = CostSensitivePastingClassifier().fit(self.x_train, self.y_train, self.cost_mat)
        
        return self.return_results(pst, self.x_train, self.x_test, self.x_val) 

    def bagging_costcla(self):

        self.cost_cla()             
        bag = CostSensitiveBaggingClassifier().fit(self.x_train, self.y_train, self.cost_mat)
        
        return self.return_results(bag, self.x_train, self.x_test, self.x_val) 

    def result(self):
        """Returns the results.
            
        Returns:
            Returns the results of the specified model.
        """
        
        if self.expt == "logistic":
            
            return self.logistic()
 
        if self.expt == "mip_logistic":
            
            if self.logistic_model is None:
                self.logistic(False)   
                
            if (self.r_train.shape[1] == 2):
                betas = np.zeros((2, self.x_train.shape[1]))
                biases = np.zeros(2)
                betas[0,] = -self.logistic_model.coef_[0]
                biases[0] = -self.logistic_model.intercept_[0]
            else:
                betas = self.logistic_model.coef_
                biases = self.logistic_model.intercept_
        
            self.cslr_model.expt = "mip_logistic"                        
            self.cslr_model.set_initial_params(betas, biases)
            
            return self.cslr_model.result()   
        
        if self.expt == "tree":
            
            return self.tree()
        
        if self.expt == "xgboost":
            
            return self.xgboost()        

        if self.expt == "svm":
            
            return self.supportvector()      

        if self.expt == "mip_svm":
            
            if self.svm_model is None:
                self.supportvector(False)   
                
            if (self.r_train.shape[1] == 2):
                betas = np.zeros((2, self.x_train.shape[1]))
                biases = np.zeros(2)
                betas[0,] = -self.svm_model.coef_
                biases[0] = -self.svm_model.intercept_
            else:
                betas = self.svm_model.coef_
                biases = self.svm_model.intercept_
                
            self.cslr_model.expt = "mip_svm"                        
            self.cslr_model.set_initial_params(betas, biases)
       
            return self.cslr_model.result()   

        if self.expt == "svm_cost":
            
            return self.svm_costsensitive()

        if self.expt == "mip_svm_cost":
            
            if self.svm_cost_model is None:
                self.svm_costsensitive(False)   
                
            if (self.r_train.shape[1] == 2):
                betas = np.zeros((2, self.x_train.shape[1]))
                biases = np.zeros(2)
                betas[0,] = -self.svm_cost_model.w
                biases[0] = -self.svm_cost_model.b                
            else:
                raise RuntimeError("You cannot run SVM-Cost algorithm on multiclass datasets.")
                
            self.cslr_model.expt = "mip_svm_cost"                        
            self.cslr_model.set_initial_params(betas, biases)
       
            return self.cslr_model.result()   
        
        if self.expt == "tree_costcla":
            
            return self.tree_costcla()

        if self.expt == "rf_costcla":
            
            return self.rf_costcla()
        
        if self.expt == "rp_costcla":
            
            return self.rp_costcla()        

        if self.expt == "logistic_costcla":
            
            return self.logistic_costcla()      

        if self.expt == "bagging_costcla":
           
            return self.bagging_costcla()

        if self.expt == "pasting_costcla":
           
            return self.pasting_costcla()
