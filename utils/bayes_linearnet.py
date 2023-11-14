import optuna
import torch
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold
from Models.linearnet import LinearNet
from Models.opt_torch import Optimization
from utils.utils import test_performance

class cslr_bayes: 
    """ Hyperparameter optimisation for CSLR using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
        device: device to store the data, cpu or cuda
        scaler: whether to scale the data
        batch_norm: whether to apply batch normalization on the network
        n_epochs: total number of epochs.
        n_steps: number of epochs to evaluate the results
        batch_size: batch-size for the network
        dnn_layers: number of layers of the network
        hidden_size: size of the hidden layers of the network
        bayes_params: bayesian optimization parameters
        bayes_trials: number of trials
        n_val_splits: number of inner-cross validation folds
    """

    def __init__(self, formatter, data, labels, returns, config):
        """Instantiates the attributes and parameters.
            
        Args:
            formatter: formatter for the specified experiment
            data: the data set
            labels: labels of the data
            returns: returns of the data
            config: configuration for the experiment
        """
        
        self.device = config.get("device", "cpu")
        self.scaler = config.get("scaler", False)          
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", self.n_epochs)        
        self.batch_norm = config.get("batch_norm", False)        
        self.batch_size = config.get("batch_size", len(data))
        self.dnn_layers = config.get("dnn_layers", 1)    
        self.hidden_size = config.get("hidden_size", [])
        
        self.formatter = formatter      
        self.bayes_params = self.formatter.get_bayes_params()
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10)            
        
        self.data = data
        self.returns = returns
        self.labels = labels
        self._max_batch = math.floor(math.log2(self.data.shape[0]))
    
    def train_bayes_linear(self, trial):                 
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'lr_rate' : trial.suggest_uniform('lr_rate', 0.00005, 0.01),
                 'batch_size': trial.suggest_int('batch_size', 7, self._max_batch), 
                 'dnn_layers': trial.suggest_int('dnn_layers', 1, 3),
                 "n_inputs" : self.data.shape[1],
                 "n_outputs" : self.returns.shape[1],
                 "batch_norm" : self.batch_norm,
                 "n_epochs": self.n_epochs,
                 "n_steps": self.n_steps
        }
        
        # specifying hidden layer size if there are more than 2 layers
        if len(self.hidden_size) != space["dnn_layers"]:
            for _ in range((space["dnn_layers"] - (1 + len(self.hidden_size)))):
                self.hidden_size.append(math.floor(math.sqrt(space["n_inputs"] + space["n_outputs"]) + 5))
        space["hidden_size"] = self.hidden_size
        
        test_return = np.zeros(self.n_val_splits)
        count = 0
        skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)
            
        for train_index, test_index in skf.split(self.data, self.labels):
                
            x_train, x_test = self.formatter.transform_inputs(self.data[train_index], self.data[test_index])
            if self.scaler == True:
                x_train, x_test = self.formatter.transform_inputs(x_train,x_test)              
            r_train, r_test = self.returns[train_index], self.returns[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            
            space["batch_size"] = (2**space['batch_size'] if space['batch_size'] < self._max_batch else len(x_train))

            # tensorize the data
            x_train_nn = torch.Tensor(x_train).to(self.device)   
            x_test_nn = torch.Tensor(x_test).to(self.device)
            r_train_nn = torch.Tensor(r_train).to(self.device)
            r_test_nn = torch.Tensor(r_test).to(self.device)            

            # build the network
            model = LinearNet(space).to(self.device)           
            optimizer = torch.optim.SGD(model.parameters(), lr=space["lr_rate"], momentum=0.8, nesterov=True)
            #optimizer = torch.optim.Adam(self.model.parameters(), lr=best_lr)

            optimization = Optimization(model, optimizer, space)
            optimization.train(x_train_nn, y_train, r_train_nn, x_test_nn, y_test, r_test_nn)

            _, _, test_probs = optimization.evaluate(x_test_nn, r_test_nn)
            test_preds = np.argmax(test_probs.detach().cpu().clone().numpy(), axis=1)
            returns = test_performance(y_test, test_preds, r_test)
            test_return[count] = returns[0]
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
        study.optimize(func=self.train_bayes_linear, n_trials=self.bayes_trials)
    
        return study.best_params

