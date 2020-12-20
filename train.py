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


"""Trains CSLR based on a defined set of parameters.
    Uses default parameters supplied from the configs file to train a TFT model from
    scratch.
    Usage:
    python3 train {expt_name} {use_cslr} {mip_wi} {mip} {ml} {hyperparam_opt}
    {time_limit} {output_folder}
    Command line args:
    expt_name: Name of dataset/experiment to train.
    use_cslr: Whether to use Cost-Sensitive Logistic Regression.
    mip_wi: Whether to use Mixed-Integer Programming with Initial Solution.
    mip: Whether to use Mixed-Integer Programming.
    ml: Whether to use state-of-the-art Machine Learning models.
    hyperparam_opt: Whether to use Bayesian Optimization for hyperparameter tuning.
    time_limit: Time limit for MIP models.
    output_folder: Root folder in which experiment is saved.
"""

import argparse

import data_formatters.base
import expt_settings.configs

from Models.cslr import cslr
from Models.ml import ml_models
from bayes_opt.utils import write_results

ExperimentConfig = expt_settings.configs.ExperimentConfig

def main(
        expt_name,
        edcs_methods,
        use_ml,
        data_formatter
    ): 
    
    """Trains cslr based on defined model params.
      Args:
        expt_name: Name of experiment.
        edcs_methods: List of EDCS solving approaches.
        use_ml: Whether to use state-of-the-art Machine Learning models.
        data_formatter: Dataset-specific data fromatter (s
        expt_settings.dataformatter.GenericDataFormatter)
    """
    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))
    
    print("*** Training from defined parameters for {} ***".format(expt_name))
    
    ml_methods = ["tree", "xgboost", "logistic"]
    
    params = data_formatter.get_default_model_params()
    for k in params:
        print("{}: {}".format(k, params[k]))  
        
    simulated_expt = ExperimentConfig.simulated_experiments
    writer = write_results(data_formatter)
    num_repeats = data_formatter.params["num_repeats"]
    
    for n in range(num_repeats):
        
        data_formatter.split_data()
        data_formatter.save_models(n, expt_name, simulated_expt)
        
        if data_formatter.params["validation"] == True:

            data_formatter.perform_validation(n)                    

            for split in range(data_formatter.n_splits):
                
                print("***Loading & splitting data***")
                train, test, valid = data_formatter.load_data(split)
                model = cslr(data_formatter, train, test, valid)

                for m in edcs_methods:
                    
                    model.expt = m
                    train_scores, test_scores, val_scores = model.result()
                    writer.collect_results(n, m, train_scores, test_scores, val_scores)

                if use_ml == "yes":
                    ml_model = ml_models(data_formatter, train, test, valid)
                    for m in ml_methods:
                        ml_model.expt = m
                        train_scores, test_scores, val_scores = ml_model.result()
                        writer.collect_results(n, m, train_scores, test_scores, val_scores)
        else:
            
            print("***Loading & splitting data***")
            train, test, _ = data_formatter.load_data()
            model = cslr(data_formatter, train, test)
            
            for m in edcs_methods:
                
                model.expt = m
                train_scores, test_scores, _ = model.result()
                writer.collect_results(n, m, train_scores, test_scores)

            if use_ml == "yes":
                ml_model = ml_models(data_formatter, train, test)
                for m in ml_methods:
                    ml_model.expt = m
                    train_scores, test_scores, _ = ml_model.result()
                    writer.collect_results(n, m,train_scores, test_scores)

        print("Printing test results.....")
        print(writer.test_df.tail(6))
        print(writer.mip_perf)
        writer.print_results()
            
    #print(writer.train_df)         
    print("Printing average results.....")
    print(writer.average_results)
    print("Printing mip performance results.....")
    print(writer.mip_avg_perf)
      
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
            "mip_wi",
            metavar="w",
            type=str,
            nargs="?",
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
            "hyperparam_opt",
            metavar="h",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="no",
            help="Whether to use Hyperparam Opt on CSLR")                
        parser.add_argument(
            "time_limit",
            metavar="tm",
            type=int,
            nargs="?",
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
    
    config = ExperimentConfig(expt_name, folder)
    formatter = config.make_data_formatter()
           
    formatter.expt_name = expt_name        
    formatter.data_folder = config.data_folder
    formatter.model_folder = config.model_folder
    formatter.results_folder = config.results_folder
    formatter.time_limit = time_limit
    formatter.bayes = (True if hyperparam_opt == "yes" else False)
    
    edcs_methods = []
    if use_cslr == "yes":
        edcs_methods.append("cslr")
    if mip_wi == "yes":
        edcs_methods.append("mip_wi")
    if mip == "yes":
        edcs_methods.append("mip")

    # For new experiments customise inputs to main() .
    main(
        expt_name=expt_name,
        edcs_methods=edcs_methods,
        use_ml=ml,
        data_formatter=formatter
    ) 

