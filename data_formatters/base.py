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

"""Default data formatting functions for experiments.
For new datasets, inherit form GenericDataFormatter and implement
all abstract functions.
These dataset-specific methods:
1) Perform the necessary input feature engineering & normalisation steps
2) Are responsible for train, validation and test splits
"""

import os
import abc
import pickle

from sklearn.model_selection import StratifiedKFold

class GenericDataFormatter(abc.ABC):
    """Abstract base class for all data formatters.
    User can implement the abstract methods below to perform dataset-specific
    manipulations.
    """

    @abc.abstractmethod
    def transform_inputs(self, train, test=None, valid=None):
        """Performs feature transformation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def split_data(self):
        """Performs the default train and test splits."""
        raise NotImplementedError()

    def perform_validation(self, n):
        """Performs validation sets."""

        fixed_params = self.get_fixed_params()
                    
        self.train = []
        self.valid = []
        self.val_set = []         
        self.train_set = []
        self.n_splits = fixed_params.get("n_splits", 10)                                            
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        
        for train_index, val_index in skf.split(self.x_train, self.y_train):
                    
            self.train_set.append(train_index)
            self.val_set.append(val_index)
            self.train.append([self.x_train[train_index],
                               self.r_train[train_index],
                               self.rmax_train[train_index],
                               self.y_train[train_index]]
                             )
            self.valid.append([self.x_train[val_index],
                               self.r_train[val_index],
                               self.rmax_train[val_index],
                               self.y_train[val_index]]
                             )

        pickle.dump(self.train_set,
                    open(self.data_path+"_train_sets.dat", "wb")
                   )
        pickle.dump(self.val_set,
                    open(self.data_path+"_valid_sets.dat", "wb")
                   )
                                                  

    def load_data(self, split=None):
        """Returns train, test and validation data for experiments."""

        train, test, valid = self.train, self.test, self.valid        
        fixed_params = self.get_experiment_params()
        
        if fixed_params['validation'] == True:
          
            train = train[split]
            valid = valid[split]
            
            if fixed_params['testing'] == False:
                
                test = valid
                valid = []
                
            self.model_path = self.model_path_temp+"_"+str(split)
        
        return train, test, valid

    def save_models(self, n, expt, simulated_expt):
        """Save the data for experiments.
        Args:
            n, expt, simulated_expt =
            # of repeat, name of the experiment, name of the simulated experiments
        """
        
        for k in simulated_expt:
            if expt == k:
                data_path = self.data_folder+self.expt_path
                model_path = self.model_folder+self.expt_path
                results_path = self.results_folder+self.expt_path
                break
            else:
                data_path = self.data_folder
                model_path = self.model_folder
                results_path = self.results_folder
                
        self.data_path = data_path+"_"+str(n)
        self.model_path = model_path+"_"+str(n)
        self.results_path = results_path
        self.model_path_temp = self.model_path
        self.results_path_temp = self.results_path 
        
        data = [self.train, self.test] 
        pickle.dump(data, open(self.data_path+".dat", "wb"))
        
    @abc.abstractmethod
    def get_fixed_params(self):
        """Defines the fixed parameters used by the model for training.
        Requires the following keys:
        to be defined...
        Returns:
          A dictionary of fixed parameters, e.g.:
          fixed_params = {
              'n_epochs': 1000,
              'device': "cpu",
              'num_repeats': 1,
              'testing': True,
              'validation': False,
              'scaler': True
          }
        """
        raise NotImplementedError 

    def get_experiment_params(self):
        """Returns fixed model parameters for experiments."""

        required_keys = [
            'n_epochs', 'num_repeats', 'device', 'testing', 'validation', 'scaler'
        ]

        fixed_params = self.get_fixed_params()

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError('Field {}'.format(k) +
                                 ' missing from fixed parameter definitions!')
                
        if fixed_params['testing'] == False and fixed_params['validation'] == False:
            raise ValueError('Please determine test or validation sets! ')               
        
        return fixed_params

    @abc.abstractmethod
    def get_tuning_params(self):
        """Defines the fixed parameters used by the model for training.
        Requires the following keys:
        to be defined....
        Returns:
          A dictionary of fixed parameters, e.g.:
          fixed_params = {
              'bayes_trials': 20,
              'batch_size_bayes': [],
              'dnn_layers_bayes': [],
              'inner_cval': True,
          }
        """
        raise NotImplementedError   

    def get_bayes_params(self):
        """Returns bayesian optimization parameters for experiments."""

        required_keys = [
            'bayes_trials', 'batch_size_bayes', 'dnn_layers_bayes', 'inner_cval'
        ]
        bayes_params = self.get_tuning_params()
        
        if self.bayes == True:
            for k in required_keys:
                if k not in bayes_params:
                    raise ValueError('Field {}'.format(k) +
                                     ' missing from bayes parameter definitions!')

        return bayes_params

