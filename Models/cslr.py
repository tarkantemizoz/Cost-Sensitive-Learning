import torch
import numpy as np
import math
import pickle
import pandas as pd

from gurobipy import *
from numpy import linalg

from utils.bayes_linearnet import cslr_bayes
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from Models.gurobi_opt import Optimization_MIP
from utils.utils import test_performance

class cslr: 
    """A class to run the proposed EDCS algorithms: CSLR, MIP-WI and MIP
        
    Attributes:
        formatter: formatter for the specified experiment
        fixed_params: fixed parameters for the specified experiment
        expt_params: experiment parameters for CSLR
        expt: name of EDCS learning model
        model: CSLR model
        time_limit: time limit for MIP-WI and MIP
        beta_initial: initial coefficients to be fed into MIP model
        analyze_first_feasible: whether to find the first feasible solution
            for MIP or MIP-WI
        first_feasible_continue: whether to continue optimizing after finding the
            first feasible solution
        num_class: number of classes of an instance
        num_features: number of features of an instance
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
        """Organizes the data for CSLR, MIP and MIP-WI
            
        Args:
            formatter: formatter for the specified experiment
            train: holds the quadruple of: (features, returns, labels)
                for train data
            test: holds the quadruple of: (features, returns, labels)
                for test data
            valid: holds the quadruple of: (features, returns, labels)
                for validation data
        """
        
        self.formatter = formatter
        self.fixed_params = self.formatter.get_experiment_params()
        self.expt_params = self.formatter.get_default_model_params()
        self.expt = "cslr"

        self.time_limit = self.formatter.time_limit
        self.beta_initial = None
        self.bias_initial = None
        self.analyze_first_feasible = False
        self.first_feasible_continue = False
        self.min_diff = 0.01
        self.score_upper_bound = 100
        self.score_lower_bound = -100
        self.model = None
        
        self.x_train, self.x_test = train[0], test[0]
        self.r_train, self.r_test = train[1], test[1]
        self.y_train, self.y_test = train[3], test[3]     
        self.x_val = (valid[0] if valid is not None and valid != [] else None)
        self.r_val = (valid[1] if valid is not None and valid != [] else None)
        self.y_val = (valid[3] if valid is not None and valid != [] else None) 
        self.num_class = self.r_train.shape[1]
        self.num_features = self.x_train.shape[1]
        self.x_train, self.x_test, self.x_val = self.format_data()

    def scores_fnc(self, betas, biases):
        """Calculate scores and differences"""
        
        scores = np.zeros((len(self.x_train), self.num_class))
        scores_diff = np.zeros((len(self.x_train),
                                ((self.num_class * (self.num_class - 1)) // 2)))
        for i in range(len(self.x_train)):
            for k in range(self.num_class):
                scores[i,k] = sum(self.x_train[i,j] * betas[k,j] for j in range(self.num_features)) + biases[k]
        diff_ind = 0
        for k in range(self.num_class-1):
            for t in range(self.num_class-(k+1)):
                scores_diff[:,diff_ind] = np.subtract(scores[:,k], scores[:,(k+t+1)])
                diff_ind += 1
        return scores, scores_diff 
  
    def opt_initial(self):
        """Removing instances that creates infeasibility for mip-wi"""
        
        params_before = self.beta_initial.copy()
        beta_scaled = self.beta_initial.copy()
        bias_before = self.bias_initial.copy()
        bias_scaled =self.bias_initial.copy()
        update = False
        # Define a tolerance for minimum score difference and bounds
        tolerance = 1e-6
        max_iterations = 1000  # Prevent endless loops
        
        for _ in range(max_iterations):
            scores, scores_diff = self.scores_fnc(beta_scaled, bias_scaled)
            score_violations_upper = np.max(scores) - self.score_upper_bound
            score_violations_lower = self.score_lower_bound - np.min(scores)
            diff_violations = self.min_diff - np.min(np.abs(scores_diff), where=(scores_diff!=0), initial=self.min_diff)

            # Check if all conditions are met
            if score_violations_upper <= 0 and score_violations_lower <= 0 and diff_violations <= tolerance:
                break

            # Adjust beta values
            if score_violations_upper > 0:
                beta_scaled *= (self.score_upper_bound - tolerance) / np.max(scores)
                bias_scaled *= (self.score_upper_bound - tolerance) / np.max(scores)         
            elif score_violations_lower > 0:
                beta_scaled *= (self.score_lower_bound + tolerance) / np.min(scores)
                bias_scaled *= (self.score_lower_bound + tolerance) / np.min(scores)              
            elif diff_violations > tolerance:
                beta_scaled *= (self.min_diff - tolerance) / np.min(np.abs(scores_diff), where=(scores_diff!=0), initial=self.min_diff)
                bias_scaled *= (self.min_diff - tolerance) / np.min(np.abs(scores_diff), where=(scores_diff!=0), initial=self.min_diff)           
            update = True
            # Check for excessive reduction, preventing beta from becoming trivial
            if np.linalg.norm(beta_scaled, ord=1) < tolerance:
                print("Warning: Beta values have been reduced too much. Adjusting strategy may be required.")
                beta_scaled = params_before
                bias_scaled = bias_before
                break    
            
        scores, scores_diff = self.scores_fnc(beta_scaled, bias_scaled)            
        self.beta_initial = beta_scaled
        self.bias_initial = bias_scaled
     
        if (update):
            print("Parameters before scaling:") 
            print(params_before) 
            print(bias_before)            
            train_preds = self.calculate_preds(params_before, bias_before, self.x_train)
            train_returns = test_performance(self.y_train, train_preds, self.r_train)[0]
            test_preds = self.calculate_preds(params_before, bias_before, self.x_test)
            test_returns = test_performance(self.y_test, test_preds, self.r_test)[0]        
            print("Train returns before scaling: {}, Test returns before scaling: {}".format(train_returns, test_returns))
            print("Parameters after scaling")  
            print(self.beta_initial)
            print(self.bias_initial)           
            train_preds = self.calculate_preds(self.beta_initial, self.bias_initial, self.x_train)
            train_returns = test_performance(self.y_train, train_preds, self.r_train)[0]
            test_preds = self.calculate_preds(self.beta_initial, self.bias_initial, self.x_test)
            test_returns = test_performance(self.y_test, test_preds, self.r_test)[0]
            print("Train returns after scaling: {}, Test returns after scaling: {}".format(train_returns, test_returns))

        is_deviate = []
        for i in range(len(scores_diff)):
            if (abs(scores_diff[i,:]) < self.min_diff).any() or (scores[i,:] > self.score_upper_bound).any() or (scores[i,:] < self.score_lower_bound).any():
                is_deviate.append(i)
                print("Infeasible instance, score is out of bounds or minimum score difference is not attained") 
        if (len(is_deviate) > 0):
            print("Number of infeasible instances: {}".format(len(is_deviate)))
                  
        self.x_train = np.delete(self.x_train, is_deviate, 0)
        self.r_train = np.delete(self.r_train, is_deviate, 0)
        self.y_train = np.delete(self.y_train, is_deviate)  

    def gradient(self):
        """Running Cost-sensitive Logistic Regression
            
        Returns:
            The function that calculates the results.
        """
        
        # setting up the model parameters
        self.dnn_layers = int(self.expt_params.get("dnn_layers", 1))
        self.hidden_size = self.expt_params.get("hidden_size", [])
        self.lr_rate = float(self.expt_params.get("learning_rate", 5e-3))   
        self.batch_size = int(self.expt_params.get("batch_size", len(self.x_train)))
        self.max_batch = math.floor(math.log2(self.x_train.shape[0]))  
        self.bayes = self.formatter.bayes
        self.model = self.expt_params.get("model", None)     
        
        self.scaler = self.fixed_params.get("scaler", False)        
        self.device = str(self.fixed_params.get("device", "cpu"))
        self.n_epochs = int(self.fixed_params.get("n_epochs", 1000))
        self.n_steps = int(self.fixed_params.get("n_steps", self.n_epochs))    
        self.batch_norm = self.expt_params.get("batch_norm", False)        

        temp_x_train = self.x_train
        kernel = "no"
        if kernel == "poly":
            self.x_train = np.power(np.matmul(self.x_train, self.x_train.T) + 1, 2)
            temp = np.zeros((len(self.x_test), len(self.x_train)))
            for j in range(len(self.x_test)):
                for i in range(len(self.x_train)):
                    temp[j,i] += (1 + np.dot(self.x_test[j], temp_x_train[i])) ** 2
            self.x_test = temp
        if kernel == "linear":
            self.x_train = np.matmul(self.x_train, self.x_train.T) 
            temp = np.zeros((len(self.x_test), len(self.x_train)))
            for j in range(len(self.x_test)):
                for i in range(len(self.x_train)):
                    temp[j,i] += np.dot(self.x_test[j], temp_x_train[i])
            self.x_test = temp
        if kernel == "rbf":
            n = self.x_train.shape[0]
            sigma = 1 / (n * self.x_train.var()) 
            temp_train = np.zeros((len(self.x_train), len(self.x_train)))
            for j in range(len(self.x_train)):
                for i in range(len(self.x_train)):
                    temp_train[j,i] += np.exp(-linalg.norm(temp_x_train[j]-temp_x_train[i])**2 / (2 * (sigma ** 2)))
            self.x_train = temp_train
            temp = np.zeros((len(self.x_test), len(self.x_train)))
            for j in range(len(self.x_test)):
                for i in range(len(self.x_train)):
                    temp[j,i] += np.exp(-linalg.norm(self.x_test[j]-temp_x_train[i])**2 / (2 * (sigma ** 2)))
            self.x_test = temp

        params = {
            "n_inputs" : self.num_features,
            "n_outputs": self.num_class,
            "batch_norm": self.batch_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "device": self.device,
            "scaler": self.scaler
            }        
        
        # applying bayesian optimization to find best parameters
        if False:            
            bayes = cslr_bayes(self.formatter, self.x_train, self.y_train, self.r_train, params)
            bayes_params = self.formatter.get_bayes_params()
            best_params_bayes = bayes.bayes()
            best_lr = (best_params_bayes.get("lr_rate", ""))
            best_batch_size = (best_params_bayes.get("batch_size" ""))
            best_dnn_layers = (best_params_bayes.get("dnn_layers" ""))
            self.dnn_layers = best_dnn_layers
            params["batch_size"] = (2**best_batch_size if best_batch_size < self.max_batch else len(self.x_train))
        # continuing with default parameters
        else:
            best_lr = self.lr_rate
            params["batch_size"] = self.batch_size
            
        params["dnn_layers"] = self.dnn_layers
        # specifying hidden layer size if there are more than 2 layers
        if len(self.hidden_size) != params["dnn_layers"]:
            for _ in range((params["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor(math.sqrt(self.num_features + self.num_class) + 5))
        params["hidden_size"] = self.hidden_size

        # tensorize the data
        x_train_nn = torch.Tensor(self.x_train).to(self.device)   
        x_test_nn = torch.Tensor(self.x_test).to(self.device)
        r_train_nn = torch.Tensor(self.r_train).to(self.device)
        r_test_nn = torch.Tensor(self.r_test).to(self.device)            

        # build the network
        self.model = LinearNet(params).to(self.device)
            
        optimizer = torch.optim.SGD(self.model.parameters(), lr=best_lr, momentum=0.8, nesterov=True)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=best_lr)

        optimization = Optimization(self.model, optimizer, params)
        
        if self.x_val is not None:
            x_val_nn = torch.Tensor(self.x_val).to(self.device)
            r_val_nn = torch.Tensor(self.r_val).to(self.device)
            optimization.train(x_train_nn, self.y_train, r_train_nn, x_val_nn, self.y_val, r_val_nn)   
        else:
            optimization.train(x_train_nn, self.y_train, r_train_nn, x_test_nn, self.y_test, r_test_nn)                
 
        #saving the model
        #torch.save(self.model.state_dict(), self.formatter.model_path+"_cslr.pt")

        if (params["dnn_layers"] == 1):         
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Extracting weights
                    if "weight" in name:
                        weights = param.data.detach().cpu().numpy()
                    # Extracting biases
                    elif "bias" in name:
                        biases = param.data.detach().cpu().numpy()           
            return self.return_results(weights, biases)
                        
        else:
            _, _, train_probs = optimization.evaluate(x_train_nn, r_train_nn)
            train_preds = np.argmax(train_probs.detach().cpu().clone().numpy(), axis=1)
            _, _, test_probs = optimization.evaluate(x_test_nn, r_test_nn)
            test_preds = np.argmax(test_probs.detach().cpu().clone().numpy(), axis=1)
            
            train_scores = test_performance(self.y_train, train_preds, self.r_train)
            test_scores = test_performance(self.y_test, test_preds, self.r_test)
            val_scores = []

            if self.x_val is not None:           
                _, _, val_probs = optimization.evaluate(x_val_nn, r_val_nn)
                val_preds = np.argmax(val_probs.detach().cpu().clone().numpy(), axis=1)          
                
                val_scores = test_performance(self.y_val, val_preds, self.r_val)
            
            return train_scores, test_scores, val_scores
    
    def mip_opt(self, initial = False):
        """Building linear programming models, mip or mip-wi, for EDCS problems
            
        Returns:
            The function that calculates the results.
        """
        
        # building the model, mip-wi or mip
        if (initial):   
            #updating the inputs in case of infeasibility
            self.opt_initial()
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
            for i in range(self.num_class):
                for j in range(self.num_features):
                    # providing the initial solution
                    self.model_mip.beta[i,j].start = self.beta_initial[i,j]
                self.model_mip.bias[i].start = self.bias_initial[i]
            self.model_mip.m.update() 
        else:
            self.model_mip = Optimization_MIP({}, self.x_train, self.r_train)
        
        self.mip_soln = {}
        self.model_mip.m.modelSense = GRB.MAXIMIZE
        self.model_mip.m.setParam(GRB.Param.TimeLimit, self.time_limit)
        min_total_returns = sum(np.amin(self.r_train, 1))
        max_total_returns = sum(np.amax(self.r_train, 1))        

        # optimizing until first feasible solution is found
        if self.analyze_first_feasible == True:
            if self.expt == "mip_wi":
                self.model_mip.m.setParam(GRB.Param.SolutionLimit, 2)
            else:
                self.model_mip.m.setParam(GRB.Param.SolutionLimit, 1)
                
            self.model_mip.m.optimize()
            mip_gap, obj, time_passed = (self.model_mip.m.MIPGap,
                                         self.model_mip.m.objVal,
                                         self.model_mip.m.RunTime)
            self.mip_soln["first_feasible"] = {
                "mip_gap": mip_gap,
                "obj_val": obj,
                "norm_obj_val": (obj-min_total_returns)/(max_total_returns - min_total_returns),
                "time_passed": time_passed
            }
            
            # continuing optimization until the time limit
            if self.first_feasible_continue == True:
                runTime = self.model_mip.m.RunTime
                new_time_limit = self.time_limit - runTime
                
                # continue if there is still time left
                if new_time_limit > 0:
                    self.model_mip.m.setParam(GRB.Param.TimeLimit, new_time_limit)
                    self.model_mip.m.setParam(GRB.Param.SolutionLimit, 2000000000)
                    self.model_mip.m.optimize()
                    runTime = runTime + self.model_mip.m.RunTime
                    mip_gap, obj, time_passed = self.model_mip.m.MIPGap, self.model_mip.m.objVal, runTime
                    self.mip_soln["last_feasible"] = {
                        "mip_gap": mip_gap,
                        "obj_val": obj,
                        "norm_obj_val": (obj-min_total_returns)/(max_total_returns - min_total_returns),
                        "time_passed": time_passed
                    }
            
                # ending the optimization if already passed the time limit
                else:
                    self.mip_soln["last_feasible"] = self.mip_soln["first_feasible"]

            # ending the optimization
            else:
                self.mip_soln["last_feasible"] = self.mip_soln["first_feasible"]
        
        # optimizing until the time limit
        else:
            self.model_mip.m.optimize()
            mip_gap, obj, time_passed = (self.model_mip.m.MIPGap,
                                         self.model_mip.m.objVal,
                                         self.model_mip.m.RunTime)
            self.mip_soln["last_feasible"] = {
                "mip_gap": mip_gap,
                "obj_val": obj,
                "norm_obj_val": (obj-min_total_returns)/(max_total_returns - min_total_returns),
                "time_passed": time_passed
            }
            self.mip_soln["first_feasible"] = self.mip_soln["last_feasible"]
       
        vars_beta = np.zeros((self.num_class, self.num_features))    
        vars_bias = np.zeros(self.num_class)
        try:
            for k in range(self.num_class):
                for j in range(self.num_features):
                    vars_beta[k,j] = self.model_mip.beta[k,j].x
                vars_bias[k] = self.model_mip.bias[k].x
            #print("Parameters after optimization")
            #print(vars_beta)
            #print(vars_bias)
            return self.return_results(vars_beta, vars_bias)       
        except:
            print("MIP parameter could not be written.")
            return self.return_results(vars_beta, vars_bias)
   
    def return_results(self, betas, biases):

        train_preds = self.calculate_preds(betas, biases, self.x_train)
        test_preds = self.calculate_preds(betas, biases, self.x_test)
        
        train_scores = test_performance(self.y_train, train_preds, self.r_train)
        test_scores = test_performance(self.y_test, test_preds, self.r_test)
        val_scores = []

        if self.x_val is not None:           
            val_preds = self.calculate_preds(betas, biases, self.x_val)
          
            val_scores = test_performance(self.y_val, val_preds, self.r_val)

        if self.expt != "cslr":
            train_scores.append(self.mip_soln)
            
        return train_scores, test_scores, val_scores

    def result(self):
        """Returns the results.
            
        Returns:
            Returns the results of the specified model.
        """
        
        if self.expt == "cslr":
            
            return self.gradient()
                
        if self.expt == "mip_wi":
            
            if (int(self.expt_params.get("dnn_layers", 1)) == 1) and self.model is not None:         
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        # Extracting weights
                        if "weight" in name:
                            weights = param.data.detach().cpu().numpy()
        
                        # Extracting biases
                        elif "bias" in name:
                            biases = param.data.detach().cpu().numpy()
                # saving the solution that can be fed into mip model
                self.set_initial_params(weights, biases)
                
                return self.mip_opt(True)  
            
            else:
                raise RuntimeError("Before calling mip-wi, train a neural network without hidden layer first.")

        if self.expt == "mip":
                    
            return self.mip_opt()
        
        if self.expt == "mip_svm" or self.expt == "mip_logistic" or self.expt == "mip_svm_cost":
            
            if self.beta_initial is not None and self.bias_initial is not None:

                return self.mip_opt(True)

            else:
                raise RuntimeError("Before calling mip with initial solution, first provide the solution.")

    def calculate_preds(self, betas, biases, data):

        scores = np.zeros((len(data), self.num_class))
        preds = np.zeros(len(data))

        for i in range(len(data)):
            for k in range(self.num_class):
                scores[i,k] = sum(data[i,j] * betas[k,j] for j in range(self.num_features)) + biases[k]
            preds[i] = np.argmax(scores[i,:])

        return preds.astype(int)
    
    def set_initial_params(self, betas, biases):
        
        self.beta_initial = betas
        self.bias_initial = biases

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