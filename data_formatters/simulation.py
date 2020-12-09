#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
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
        self.seed = int(self.simul_params.get("seed", random.randint(0,10000)))
        self.n = int(self.simul_params["n"])
        self.n_test = int(self.simul_params["n_test"])
        self.noise = float(self.simul_params["noise"])
        self.num_features = int(self.simul_params["num_features"])
        self.num_class = int(self.simul_params["num_class"])
        self.expt_path = "_"+str(self.simul_params["n"])+"_"+str(self.simul_params["noise"])                    
        random.seed(self.seed)        
       
        self.x_train = np.zeros((self.n, self.num_features))
        self.x_test = np.zeros((self.n_test, self.num_features))        
        for i in range(self.n):
            for j in range(self.num_features):
                    self.x_train[i,j] = (np.random.normal(0,1) if j % 2 == 0 else random.randint(0, 1))
        for i in range(self.n_test):
            for j in range(self.num_features):
                    self.x_test[i,j] = (np.random.normal(0,1) if j % 2 == 0 else random.randint(0, 1))  
                    
        self.y_train = self.decision(self.x_train)
        self.y_test = self.decision(self.x_test)
        mean = self.y_train.mean()
        std = self.y_train.std()
        self.y_train = (self.y_train - mean) / std
        self.y_test = (self.y_test - mean) / std
        for i in range(len(self.y_train)):
            for j in range(self.num_class):
                self.y_train[i,j] = self.y_train[i,j] + np.random.normal(0,self.noise)
                
        self.r_train, self.y_train = self.returns(self.x_train, self.y_train)
        self.r_test, self.y_test =  self.returns(self.x_test, self.y_test)
        
        self.rmax_train = np.zeros(len(self.r_train))
        for i in range(len(self.r_train)):
            self.rmax_train[i] = self.r_train[i,np.argmax(self.r_train[i])]
        self.rmax_test = np.zeros(len(self.r_test))
        for i in range(len(self.r_test)):
            self.rmax_test[i] = self.r_test[i,np.argmax(self.r_test[i])]                
            
    def returns(self, arr, y):        
        
        returns = np.zeros((len(arr),self.num_class))
        outcomes = np.zeros(len(arr))

        if self.expt_name == "ex1":
            
            for i in range(len(arr)):             
                
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(100,500)
                                if np.argmax(y[i]) == 0 else -random.uniform(50,150))    
                returns[i,1] = (random.uniform(500,2000)
                                if np.argmax(y[i]) == 1 else -random.uniform(50,150))  
                
        elif self.expt_name == "ex2":
            
            for i in range(len(arr)):
                
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(100,500)
                                if np.argmax(y[i]) == 0 else -random.uniform(50,150))    
                returns[i,1] = (random.uniform(500,2000)
                                if np.argmax(y[i]) == 1 else -random.uniform(50,150))   
                
        elif self.expt_name == "ex3":
            
            for i in range(len(arr)):
                    
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(500,2000)
                                if np.argmax(y[i]) == 0 else -random.uniform(50,150))
                returns[i,1] = (random.uniform(300,500)
                                if np.argmax(y[i]) == 1 else -random.uniform(50,150))
                returns[i,2] = (random.uniform(100,300)
                                if np.argmax(y[i]) == 2 else -random.uniform(50,150))  
                    
        elif self.expt_name == "ex4":
            
            for i in range(len(arr)):
                
                outcomes[i] = np.argmax(y[i])
                returns[i,0] = (random.uniform(500,2000)
                                if np.argmax(y[i]) == 0 else -random.uniform(50,150))
                returns[i,1] = (random.uniform(300,500)
                                if np.argmax(y[i]) == 1 else -random.uniform(50,150))
                returns[i,2] = (random.uniform(100,300)
                                if np.argmax(y[i]) == 2 else -random.uniform(50,150)) 
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
            'n_epochs': 10000,
            'device': "cpu",
            'num_repeats': 10,
            'testing' : True,
            'validation': True,
            'scaler': False
        }
        
        if fixed_params["testing"] == False:
            raise ValueError('Testing should be set True for simulations!') 
            
        return fixed_params
    
    def get_default_model_params(self):
        """Returns default model parameters."""

        model_params = {
            'dnn_layer': 1,
            'learning_rate': 0.005,
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
            'noise': 1,
            'n': 1000,
            'n_test': 20000,
        }
        params_simul['ex2'] = {
            'num_class': 2,
            'num_features': 25,
            'noise': 1,
            'n': 1000,
            'n_test': 20000
        }
        params_simul['ex3'] = {
            'num_class': 3,
            'num_features': 15,
            'noise': 1,
            'n': 500,
            'n_test': 20000
        }
        params_simul['ex4'] = {
            'num_class': 3,
            'num_features': 25,
            'noise': 2,
            'n': 1000,
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
