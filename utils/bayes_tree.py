import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from utils.utils import test_performance

class bayes_tree: 
    """ Hyperparameter optimisation for Decision Trees using Bayesian Optimization.
        
    Attributes:
        formatter: formatter for the specified experiment
        data: the data set
        labels: labels of the data
        returns: returns of the data
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
        self.bayes_params = self.formatter.get_bayes_params()    
        self.bayes_trials = self.bayes_params["bayes_trials"]
        self.n_val_splits = self.bayes_params.get("n_val_splits", 10) 

        self.data = data
        self.labels = labels  
        self.returns = returns    

    def train_bayes_tree(self, trial):
        """Applying bayesian optimization trials
            
        Args:
            trial: bayesian optimization trial
            
        Returns:
            mean total returns
        """
        
        # setting up the search space
        space = {'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'ccp_alpha': trial.suggest_uniform('ccp_alpha', 0.0, 0.2),
                'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"])
                }
        
        test_return = np.zeros(self.n_val_splits)
        count = 0
        skf = StratifiedKFold(self.n_val_splits, shuffle=True, random_state=self.formatter.seed)

        for train_index, test_index in skf.split(self.data, self.labels):
                
            x_train, x_test = self.data[train_index], self.data[test_index]
            _, r_test = self.returns[train_index], self.returns[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
                
            dt = DecisionTreeClassifier(min_samples_leaf = space['min_samples_leaf'],
                                        ccp_alpha = space['ccp_alpha'],
                                        max_depth = space['max_depth'],
                                        criterion = space['criterion']).fit(x_train, y_train) 
            preds = dt.predict(x_test)
            
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
        study.optimize(func=self.train_bayes_tree, n_trials=self.bayes_trials)
    
        return study.best_params

