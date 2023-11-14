import data_formatters.base

import os
import feather
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

GenericDataFormatter = data_formatters.base.GenericDataFormatter

class creditcard(GenericDataFormatter):
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
 
        os.chdir('/Users/20214773/source/repos/CSLR-Revised/datasets/creditdata')
        directory = '/Users/20214773/source/repos/CSLR-Revised/datasets/creditdata'

        self.data = []
        for filename in os.listdir(directory):
            DF = feather.read_dataframe(filename)
            self.data.append(DF)

        self.seed = 30
        self.train = []
        self.test = []     
        self.valid = []      
        
    def split_data(self):
        """Split Data: train and test.
        """
        (self.x_train, self.y_train, self.r_train, self.rmax_train) = (self.data[self.seed].to_numpy(),
                                                                       self.data[self.seed+2].to_numpy().astype(int),
                                                                       self.data[self.seed+1].to_numpy(),
                                                                       np.amax(self.data[self.seed+1].to_numpy(),1))        
        (self.x_test, self.y_test, self.r_test, self.rmax_test) = (self.data[self.seed-30].to_numpy(),
                                                                   self.data[self.seed+2-30].to_numpy().astype(int),
                                                                   self.data[self.seed+1-30].to_numpy(),
                                                                   np.amax(self.data[self.seed+1-30].to_numpy(),1))   
        self.train = [self.x_train, self.r_train, self.rmax_train, self.y_train]
        self.test = [self.x_test, self.r_test, self.rmax_test, self.y_test]
        self.seed += 3

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

        fixed_params = {
            'n_epochs': 1000,
            'n_steps': 10,
            'device': "cpu",
            'num_repeats': 3, # do not change this
            'testing' : True, # do not change this
            'validation': False, # do not change this
            'scaler': False
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default model parameters."""

        model_params = {
            'dnn_layers': 1,
            'learning_rate': 0.01,
            'batch_size': 5120,
            'batch_norm': False
        }

        return model_params

    def get_tuning_params(self):
        """Returns default model parameters."""

        bayes_params = {
            'bayes_trials': 100,      
            'n_val_splits': 10
        }

        return bayes_params

