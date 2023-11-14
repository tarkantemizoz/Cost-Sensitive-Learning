import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import svm

from utils.utils import test_performance

class bayes_svm: 
    """ Hyperparameter optimisation for Support Vector Machines using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
        scaler: whether to scale the data
        bayes_params: bayesian optimization parameters
        bayes_trials: number of trials
        n_val_splits: number of inner-cross validation folds
    """
    
    def __init__(self, formatter, data, labels, returns):    
        """Instantiates the attributes and parameters.
            
        Args:
            formatter: formatter for the specified experiment
            data: the data set
            labels: labels of the data
            returns: returns of the data
        """

        self.formatter = formatter      
        self.scaler = formatter.get_experiment_params().get("scaler", False)
        self.bayes_params = self.formatter.get_bayes_params()    
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10)

        self.data = data
        self.labels = labels  
        self.returns = returns    

    def train_bayes_svm(self, trial):
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'C': trial.suggest_int('C', -2, 15)}
         
        # apply inner-cross validation         
        test_return = np.zeros(self.n_val_splits)
        count = 0
        skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)
            
        for train_index, test_index in skf.split(self.data, self.labels):
                
            x_train, x_test = self.data[train_index], self.data[test_index]
            if self.scaler == True:
                x_train, x_test = self.formatter.transform_inputs(x_train,x_test)  
            _, r_test = self.returns[train_index], self.returns[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            svm_model = svm.SVC(kernel = 'linear', C = 2 ** space['C']).fit(x_train, y_train)
            preds = svm_model.predict(x_test).astype(int)
            
            returns = test_performance(y_test, preds, r_test)
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
        study.optimize(func=self.train_bayes_svm, n_trials=self.bayes_trials)
    
        return study.best_params

