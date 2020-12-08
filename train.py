#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore")
import argparse
import sys


import data_formatters.base
import expt_settings.configs
from Models.cslr import cslr
from Models.ml import ml_models
from bayes_opt.utils import write_results

ExperimentConfig = expt_settings.configs.ExperimentConfig

def main(
        expt_name,
        use_cslr,
        use_mip_wi,
        use_mip,
        use_ml,
        use_hyperparam_opt,
        time_limit,
        data_formatter
    ): 
    
    """Trains tft based on defined model params.
      Args:
        expt_name: Name of experiment
        data_formatter: Dataset-specific data fromatter (see
          expt_settings.dataformatter.GenericDataFormatter)
    """
    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))
    
    print("*** Training from defined parameters for {} ***".format(expt_name))
    
    ml_methods = ["logistic", "tree", "xgboost"]
    
    params = data_formatter.get_default_model_params()
    for k in params:
        print("{}: {}".format(k, params[k]))  
        
    data_formatter.bayes = (True if use_hyperparam_opt == "yes" else False)    
    simulated_expt = ExperimentConfig.simulated_experiments
    writer = write_results(data_formatter)
    num_repeats = data_formatter.params["num_repeats"]
    
    for n in range(num_repeats):
        
        data_formatter.split_data()
        data_formatter.save_models(n, expt_name, simulated_expt)
        
        if data_formatter.params["validation"] == True:

            data_formatter.perform_validation(n)                    

            for split in range(data_formatter.n_splits):
                
                print("Loading & splitting data...")                       
                train, test, valid = data_formatter.load_data(split)
                model = cslr(data_formatter, 
                             train,
                             test,
                             valid
                            )
                if use_cslr == "yes":
                    train_scores, test_scores, val_scores = model.gradient()
                    writer.collect_results(n,
                                           "cslr",
                                           train_scores,
                                           test_scores,
                                           val_scores
                                          )
                if use_mip_wi == "yes":
                    model.time_limit = time_limit
                    model.give_initial = True 
                    train_scores, test_scores, val_scores = model.mip_opt()                   
                    writer.collect_results(n,
                                           "mip_wi",
                                           train_scores,
                                           test_scores,
                                           val_scores
                                          )                    
                if use_mip == "yes":
                    model.time_limit = time_limit                
                    model.give_initial = False                                  
                    train_scores, test_scores, val_scores = model.mip_opt()
                    writer.collect_results(n,
                                           "mip",
                                           train_scores,
                                           test_scores,
                                           val_scores
                                          )                    
                if use_ml == "yes":
                    ml_model = ml_models(data_formatter, 
                                         train,
                                         test,
                                         valid
                                        )
                    for m in ml_methods:
                        ml_model.expt = m
                        train_scores, test_scores, val_scores = ml_model.result()
                        writer.collect_results(n,
                                               m,
                                               train_scores,
                                               test_scores,
                                               val_scores
                                              )                        
        else:
            print("Loading & splitting data...")                                    
            train, test, _ = data_formatter.load_data()                        
            model = cslr(data_formatter, 
                         train,
                         test
                        )
            if use_cslr == "yes":
                train_scores, test_scores, _ = model.gradient()
                writer.collect_results(n,
                                       "cslr",
                                       train_scores,
                                       test_scores
                                       )       
            if use_mip_wi == "yes":
                model.time_limit = time_limit                  
                model.give_initial = True              
                train_scores, test_scores, _ = model.mip_opt()
                writer.collect_results(n,
                                       "mip_wi",
                                       train_scores,
                                       test_scores
                                       )                  
            if use_mip == "yes":
                model.time_limit = time_limit                  
                model.give_initial = False                                  
                train_scores, test_scores, _ = model.mip_opt()
                writer.collect_results(n,
                                       "mip",
                                       train_scores,
                                       test_scores
                                       )                  
            if use_ml == "yes":
                ml_model = ml_models(data_formatter, 
                                     train,
                                     test
                                    )
                for m in ml_methods:
                    ml_model.expt = m
                    train_scores, test_scores, _ = ml_model.result()
                    writer.collect_results(n,
                                           m,
                                           train_scores,
                                           test_scores
                                           )                           
        print("Printing test results.....")
        print(writer.test_df)
    #print(writer.train_df)         
    writer.print_results()
    print("Printing average results.....")
    print(writer.average_results)  
      
if __name__ == "__main__":

    def get_args():
        """Gets settings from command line."""

        datasets = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description = "Dataset and experiment configs")
        
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="ex1",
            choices=datasets,
            help="Dataset Name. Default={}".format(",".join(datasets)))            
        parser.add_argument(
            "cslr",
            metavar="c",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="yes",
            help="Whether to use Cost Sensitive Logistic Learning") 
        parser.add_argument(
            'mip_wi',
            metavar='w',
            type=str,
            nargs='?',
            choices=["yes", "no"],            
            default="no",
            help="Whether to use MIP-WI")
        parser.add_argument(
            "mip",
            metavar="m",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="no",
            help="Whether to use MIP")        
        parser.add_argument(
            "ml",
            metavar="ml",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="no",
            help="Whether to use ML algorithms")                            
        parser.add_argument(
           'hyperparam_opt',
            metavar='h',
            type=str,
            nargs='?',
            choices=["yes", "no"],            
            default="no",
            help="Whether to use Hyperparam Opt on CSLR")                
        parser.add_argument(
            'time_limit',
            metavar='tm',
            type=int,
            nargs='?',
            default=100,
            help="Time Limit")               
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for results and models")
        
        args = parser.parse_known_args()[0]
        
        folder = None if args.output_folder == "." else args.output_folder

        return (args.expt_name,
                args.cslr,
                args.mip_wi,
                args.mip,
                args.ml,
                args.hyperparam_opt,
                args.time_limit,
                folder
               )

    expt_name, use_cslr, mip_wi, mip, ml, hyperparam_opt, time_limit, folder = get_args()
    
    print("Using output folder {}".format(folder))    
    config = ExperimentConfig(expt_name, folder)
    formatter = config.make_data_formatter()
           
    formatter.expt_name = expt_name        
    formatter.data_folder = config.data_folder
    formatter.model_folder = config.model_folder
    formatter.results_folder = config.results_folder
    
    main(
        expt_name=expt_name,
        use_cslr=use_cslr,
        use_mip_wi=mip_wi,
        use_mip=mip,
        use_ml=ml,
        use_hyperparam_opt=hyperparam_opt,
        time_limit=time_limit,
        data_formatter=formatter
    ) 

