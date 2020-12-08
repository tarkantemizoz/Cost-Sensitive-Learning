#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import data_formatters.simulation
import data_formatters.bank

class ExperimentConfig(object):
    
    """Defines experiment configs and paths to outputs.
       Attributes:
        root_folder: Root folder to contain all experimental outputs.
        experiment: Name of experiment to run.
        data_folder: Folder to store data for experiment.
        model_folder: Folder to store models.
        results_folder: Folder to store results.
    """
    simulated_experiments = ['ex1', 'ex2', 'ex3', 'ex4']
    default_experiments = ['bank_credit', 'ex1', 'ex2', 'ex3', 'ex4']

    def __init__(self, experiment='ex1', root_folder=None):
        
          
        """Creates configs based on default experiment chosen.
        Args:
          experiment: Name of experiment.
          root_folder: Root folder to save all outputs of training.
        """

        if experiment not in self.default_experiments:
            
            raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
        if root_folder is None:
            
            root_folder = os.path.join(
                 os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
            print('Using root folder {}'.format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, 'data', experiment)
        self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
        self.results_folder = os.path.join(root_folder, 'results', experiment)
        
    # Creates folders if they don't exist
        for relevant_directory in [
            
            self.root_folder, self.data_folder, self.model_folder,
            self.results_folder
            ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

        self.data_folder = os.path.join(self.data_folder, experiment)
        self.model_folder = os.path.join(self.model_folder, experiment)
        self.results_folder = os.path.join(self.results_folder, experiment)


    def make_data_formatter(self):
        """Gets a data formatter object for experiment.
        Returns:
          Default DataFormatter per experiment.
        """
        dataset = {       
            'bank_credit': data_formatters.bank.bank_credit
        }
        for ex in ExperimentConfig.simulated_experiments:
            dataset[ex] = data_formatters.simulation.data_generator
        
        return dataset[self.experiment]()

