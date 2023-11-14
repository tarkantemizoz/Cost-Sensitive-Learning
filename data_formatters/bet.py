import numpy as np
import pickle
import pandas as pd

import data_formatters.base

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

GenericDataFormatter = data_formatters.base.GenericDataFormatter

class betting(GenericDataFormatter):
    """A helper class to simulate data for Cost Sensitive Learning"
        
        Attributes:
        params: experiment parameters
        scaler: whether to apply scaling
        validation: whether to have seperate validation folds
        testing: whether to have testing data
        features: features of the bank credit data
        returns: returns of the bank credit data
        outcomes: labels of the bank credit data
        rmax: maximum returns of the bank credit data
        seed: seed to generate the data set
        train: training data consisting of features, returns, maximum returns, labels
        validation: validation data consisting of features, returns, maximum returns, labels
        test: testing data consisting of features, returns, maximum returns, labels
        """
    
    def __init__(self):
        """Reads the data and initializes the experiment"""
        
        self.params = self.get_experiment_params()
        self.scaler = self.params.get("scaler", False)
        self.validation = self.params["validation"]
        self.testing = self.params["testing"]
        #[:,(0,8,9,10,11,12,13,14,22,23,24,25,26,27,28,36,37,38,39,40,41,42,50,51,52,53,54,55,56,64,65,66,67,68,69,70,78,70,80,81,82,83)]
        self.features = pd.read_csv('datasets/features_last_eu.csv', delimiter=' ',header=None).to_numpy()
        self.returns = pd.read_csv('datasets/realized_returns_eu.csv', delimiter=' ',header=None).to_numpy()
        self.outcomes = np.argmax(self.returns, 1)
        self.rmax = np.amax(self.returns, 1)
        self.seed = 42
        self.train = []
        self.test = []
        self.valid = []
    
    def split_data(self):
        """Split Data: train and test.
            """
        
        if self.testing == True:
            
            (self.x_train,
             x_test,
             self.y_train,
             y_test,
             self.r_train,
             r_test,
             self.rmax_train,
             rmax_test) = train_test_split(self.features,
                                           self.outcomes,
                                           self.returns,
                                           self.rmax,
                                           test_size=0.2,
                                           random_state=self.seed)
            self.test = [x_test, r_test, rmax_test, y_test]
        
        else:
            
            (self.x_train, self.y_train, self.r_train, self.rmax_train) = (self.features,
                                                                           self.outcomes,
                                                                           self.returns,
                                                                           self.rmax)
        
        self.train = [self.x_train, self.r_train, self.rmax_train, self.y_train]
        self.seed += 1


    def transform_inputs(self, train, test=None, valid=None):
        """Performs feature transformations.
            This includes standardization of data.
            
            Args:
            train, test, valid: Data to transform.
            
            Returns:
            Transformed data.
            """
            
        if self.scaler == False:
            raise ValueError('Scaler has not been set!')
                
        scaler = MinMaxScaler()
        train_scl = scaler.fit_transform(train)
                
        if valid is not None:
            test_scl = scaler.transform(test)
            valid_scl = scaler.transform(valid)
            return train_scl, test_scl, valid_scl
                        
        elif test is not None:
            test_scl = scaler.transform(test)
            return train_scl, test_scl
                                
        else:
            return train_scl

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""
        t = 0
        fixed_params = {
                'n_epochs': 100,
                'n_steps': 10,
                'device': "cpu",
                'num_repeats': 1,
                'testing' : (True if t == 0 else False),
                'validation': (False if t == 0 else True),
                'n_splits': 10,
                'scaler': True
        }
            
        return fixed_params

    def get_default_model_params(self):
        """Returns default model parameters."""
            
        model_params = {
                'dnn_layers': 1,
                'batch_size' : 32,
                'learning_rate': 0.005,
                'batch_norm': False
        }
            
        return model_params

    def get_tuning_params(self):
        """Returns default model parameters."""
        
        bayes_params = {
            'bayes_trials': 100,
            'n_val_splits': 5
        }
        
        return bayes_params