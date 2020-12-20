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

import numpy as np
import pickle
import data_formatters.base

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

GenericDataFormatter = data_formatters.base.GenericDataFormatter

class bank_credit(GenericDataFormatter):
    
    def __init__(self):
        
        self.params = self.get_experiment_params()
        self.scaler = self.params.get("scaler", False)
        self.validation = self.params["validation"]
        self.testing = self.params["testing"]
        
        self.features = pickle.load(open("datasets/bank_features","rb"))
        self.returns = pickle.load(open("datasets/bank_returns","rb"))
        self.outcomes = np.argmax(self.returns, 1)
        self.rmax = np.amax(self.returns, 1)
        self.seed = 99
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
        
        print("Scaling the data")        
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
            'n_epochs': 100,
            'device': "cpu",
            'num_repeats': 3,
            'testing' : False,
            'validation': True,
            'n_splits': 10,
            'scaler': True
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default model parameters."""

        model_params = {
            'dnn_layers': 1,
            'learning_rate': 0.0005,
            'batch_size': 64,
            'batch_norm': False
        }

        return model_params

    def get_tuning_params(self):
        """Returns default model parameters."""

        bayes_params = {
            'bayes_trials': 100,
            'batch_size_bayes': [8, 11],
            'dnn_layers_bayes': None,           
            'inner_cval': True,
            'n_val_splits': 5
        }

        return bayes_params
