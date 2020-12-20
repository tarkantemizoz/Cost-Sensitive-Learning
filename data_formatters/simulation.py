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
import data_formatters.base

GenericDataFormatter = data_formatters.base.GenericDataFormatter

class data_generator(GenericDataFormatter):
    """ A helper class to simulate data for Cost Sensitive Learning"""
    
    def __init__(self):

        self.expt_name = 'ex1'
        self.params = self.get_experiment_params()
        self.scaler = False
        self.validation = self.params["validation"]
        self.testing = self.params["testing"]
        self.seed = 42
        self.train = []
        self.test = []     
        self.valid = []  
        
    def split_data(self):

        self.create_dataset()
              
        self.test = [self.x_test, self.r_test, self.rmax_test, self.y_test]       
        self.train = [self.x_train, self.r_train, self.rmax_train, self.y_train]  
    
    def transform_inputs(self, train, test=None, valid=None):
        
        """Performs feature transformations.
        This includes standardization of data.
        Args:
          train, test, valid: Data to transform.
        Returns:
          Transformed data.
        """
        
        print("!!! The simulated data have already been scaled !!!")
        if valid is not None:
            return train, test, valid   
        elif test is not None:
            return train, test
        else:
            return train

    def create_dataset(self):
        
        self.simul_params = self.get_simulation_params()[self.expt_name]
        self.n = int(self.simul_params["n"])
        self.n_test = int(self.simul_params["n_test"])
        self.noise = float(self.simul_params["noise"])
        self.num_features = int(self.simul_params["num_features"])
        self.num_class = int(self.simul_params["num_class"])
        self.expt_path = ("_"+str(self.n)
                          +"_"+str(self.noise)
                          +"_"+str(self.time_limit))
        n_total = self.n + self.n_test
        
        np.random.seed(self.seed)
        data_int = np.random.randint(2, size = (n_total, self.num_features))
        np.random.seed(self.seed)
        data_random = np.random.normal(0, 1, size = (n_total, self.num_features))
        data = np.zeros((n_total, self.num_features))
        for j in range(self.num_features):
            data[:,j] = (data_random[:,j] if j % 2 == 0 else data_int[:,j])
        self.x_train = data[0:self.n]
        self.x_test = data[self.n:n_total]
        
        self.y_train = self.decision(self.x_train)
        self.y_test = self.decision(self.x_test)

        mean = self.y_train.mean()
        std = self.y_train.std()
        self.y_train = (self.y_train - mean) / std
        self.y_test = (self.y_test - mean) / std
        np.random.seed(self.seed + n_total)
        noise = np.random.normal(0, self.noise, size = (self.n, self.num_class))
        self.y_train = self.y_train + noise
                
        self.r_train, self.y_train = self.returns(self.y_train)
        self.r_test, self.y_test = self.returns(self.y_test)
        self.rmax_train = np.amax(self.r_train, 1)
        self.rmax_test = np.amax(self.r_test, 1)
        self.seed += 1

    def returns(self, y):
        
        returns = np.zeros((len(y),self.num_class))
        outcomes = np.argmax(y, 1)
        
        if self.expt_name == "ex1" or self.expt_name == "ex2":
            
            np.random.seed(self.seed)
            ret = np.concatenate((np.random.uniform(100,500,(len(y),1)),
                                  -np.random.uniform(50,150,(len(y),1)),
                                  np.random.uniform(500,2000,(len(y),1)),
                                  -np.random.uniform(50,150,(len(y),1))),
                                 axis = 1)
            
            for i in range(len(y)):
                
                returns[i,0] = (ret[i,0] if outcomes[i] == 0 else ret[i,1])
                returns[i,1] = (ret[i,2] if outcomes[i] == 1 else ret[i,3])
                
        elif self.expt_name == "ex3" or self.expt_name == "ex4":
            
            np.random.seed(self.seed)
            ret = np.concatenate((np.random.uniform(500,2000,(len(y),1)),
                                  -np.random.uniform(50,150,(len(y),1)),
                                  np.random.uniform(300,500,(len(y),1)),
                                  -np.random.uniform(50,150,(len(y),1)),
                                  np.random.uniform(100,300,(len(y),1)),
                                  -np.random.uniform(50,150,(len(y),1))),
                                 axis = 1)
                
            for i in range(len(y)):
            
                returns[i,0] = (ret[i,0] if outcomes[i] == 0 else ret[i,1])
                returns[i,1] = (ret[i,2] if outcomes[i] == 1 else ret[i,3])
                returns[i,2] = (ret[i,4] if outcomes[i] == 2 else ret[i,5])

        else:
            
            raise ValueError('Unknown experiment has been chosen!') 
                                                    
        return returns, outcomes.astype(int)     
    
    def decision(self, arr):        
        
        y = np.zeros((len(arr),self.num_class))
        
        if self.expt_name == "ex1":
            
            for i in range(len(arr)):     
                         
                y[i,0] = (sum(arr[i,(0,1,7,10,13)])
                          + (np.where(arr[i,6] > -0.6, 5, 0)
                             + abs(arr[i,2] * arr[i,4])))
                y[i,1] = sum(arr[i,(0,1,7,10,13)]) + abs(arr[i,12] * arr[i,14])       
                
        elif self.expt_name == "ex2":
            
            for i in range(len(arr)):
                
                    y[i,0] = (sum(arr[i,(0,1,6,9,11,16,23)])
                              + (np.where(arr[i,10] > -0.6, 5, 0)
                                 + abs(arr[i,18] * arr[i,20]) + arr[i,23]))
                    y[i,1] = (sum(arr[i,(0,1,6,9,11,16,23)])
                              + abs(arr[i,12] * arr[i,14]) + arr[i,19]) 
                
        elif self.expt_name == "ex3":
            
            for i in range(len(arr)):
                    
                    y[i,0] = sum(arr[i,(0,1,7,10,13)]) + 2
                    y[i,1] = (sum(arr[i,(0,1,7,10,13)])
                              + (np.where(arr[i,8] > 0, 7, 0)
                                 + abs(arr[i,0] * arr[i,12])))
                    y[i,2] = (sum(arr[i,(0,1,7,10,13)])
                              + (np.where(arr[i,6] > -0.2, 7, 0)
                                 + abs(arr[i,2] * arr[i,14])))
                    
        elif self.expt_name == "ex4":
            
            for i in range(len(arr)):
                
                    y[i,0] = sum(arr[i,(0,1,6,9,11,16,23)])  + 2
                    y[i,1] = (sum(arr[i,(0,1,6,9,11,16,23)])
                              + (np.where(arr[i,22] > 0, 7, 0)
                                 + abs(arr[i,2] * arr[i,14]) + arr[i,21]))
                    y[i,2] = (sum(arr[i,(0,1,6,9,11,16,23)])
                              + (np.where(arr[i,10] > -0.2, 7, 0)
                                 + abs(arr[i,12] * arr[i,20]) + arr[i,19]))                
                
        else:
            
            raise ValueError('Unknown experiment has been chosen!') 
                    
        return y   
    
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'n_epochs': 1000,
            'device': "cpu",
            'num_repeats': 1,
            'testing' : True,
            'validation': False,
            'scaler': False
        }
        
        if fixed_params["testing"] == False:
            raise ValueError('Testing should be set True for simulations!') 
            
        return fixed_params
    
    def get_default_model_params(self):
        """Returns default model parameters."""

        model_params = {
            'dnn_layer': 1,
            'learning_rate': 0.01,
            'batch_size': 1000,
            'batch_norm': False
        }
        return model_params

    def simulation_params(self):
        """Returns default simulation parameters."""

        params_simul = {}
        params_simul['ex1'] = {
            'num_class': 2,
            'num_features': 15,
            'noise': 0.5,
            'n': 1000,
            'n_test': 20000,
        }
        params_simul['ex2'] = {
            'num_class': 2,
            'num_features': 25,
            'noise': 1,
            'n': 500,
            'n_test': 20000
        }
        params_simul['ex3'] = {
            'num_class': 3,
            'num_features': 15,
            'noise': 0.5,
            'n': 5000,
            'n_test': 20000
        }
        params_simul['ex4'] = {
            'num_class': 3,
            'num_features': 25,
            'noise': 1,
            'n': 5000,
            'n_test': 20000
        }             
                                
        if params_simul[self.expt_name] is None:
                raise ValueError('Unknown experiment has been chosen! Set your experiment parameters!') 
                
        return params_simul                               
                                
    def get_simulation_params(self):
                                
        """Checks simulation parameters for experiments."""

        required_keys = [
            'num_class', 'num_features', 'noise', 'n', 'n_test'
        ]
                                        
        experiment_params = self.simulation_params()
                                
        for ex in experiment_params:
            params = experiment_params[ex]
            for k in required_keys:
                if k not in params:
                    raise ValueError('Field {}'.format(k) +
                                     ' missing from simulation parameter definitions!')
                    
        return experiment_params                                
         
    def get_tuning_params(self):
        """Returns default model parameters."""

        bayes_params = {
            'bayes_trials': 5,
            'batch_size_bayes': None,
            'dnn_layers_bayes': None,           
            'inner_cval': True,
        }

        return bayes_params
