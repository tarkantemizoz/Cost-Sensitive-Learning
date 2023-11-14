"""Trains CSLR based on a defined set of parameters.
    Uses default parameters supplied from the configs file to train a EDCS model from
    scratch.
Usage:
    python -m train {expt_name} {edcs} {ml} {cost_cla} {hyperparam_opt} {time_limit} {output_folder}
Command line args:
    expt_name: Name of dataset/experiment to train.
    edcs: Whether to use EDCS learning algorithms.
    ml: Whether to use state-of-the-art Machine Learning models.
    cost_cla: Whether to use Costcla algorithms.
    hyperparam_opt: Whether to use Bayesian Optimization for hyperparameter tuning.
    time_limit: Time limit for MIP models.
    output_folder: Root folder in which experiment is saved.
"""

import argparse

import data_formatters.base
import expt_settings.configs

from Models.cslr import cslr
from Models.ml import ml_models
from utils.utils import write_results

ExperimentConfig = expt_settings.configs.ExperimentConfig

def main(
        expt_name,
        edcs_methods,
        ml_methods,
        data_formatter
    ):
    """Trains cslr based on defined model params.
        
      Args:
        expt_name: Name of experiment.
        edcs_methods: List of EDCS solving approaches.
        ml_methods: List of machine learning approaches.
        data_formatter: Dataset-specific data fromatter.
    """
    
    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))
    
    print("*** Training from defined parameters for {} ***".format(expt_name))

    params = data_formatter.get_fixed_params()
    for k in params:
        print("{}: {}".format(k, params[k])) 
        
    params = data_formatter.get_default_model_params()
    for k in params:
        print("{}: {}".format(k, params[k]))  
        
    simulated_expt = ExperimentConfig.simulated_experiments
    writer = write_results(data_formatter)
    num_repeats = data_formatter.params["num_repeats"]

    # running the experiments
    for n in range(num_repeats):
        
        data_formatter.split_data()
        data_formatter.save_models(n, expt_name, simulated_expt)
        
        if data_formatter.params["validation"] == True:

            data_formatter.perform_validation()                    

            for split in range(data_formatter.n_splits):             
                print("***Loading & splitting data***")
                train, test, valid = data_formatter.load_data(split)
                
                # running edcs learning methods and printing the results
                model = cslr(data_formatter, train, test, valid)
                for m in edcs_methods:
                    model.expt = m
                    train_scores, test_scores, val_scores = model.result()
                    writer.collect_results(n, m, train_scores, test_scores, val_scores)
                
                # running other machine learning methods and printing the results
                ml_model = ml_models(data_formatter, train, test, valid)
                for m in ml_methods:
                    ml_model.expt = m
                    train_scores, test_scores, val_scores = ml_model.result()
                    writer.collect_results(n, m, train_scores, test_scores, val_scores)
                        
        else:
            print("***Loading & splitting data***")
            train, test, _ = data_formatter.load_data()
            
            # running edcs learning methods and printing the results
            model = cslr(data_formatter, train, test)
            for m in edcs_methods:
                model.expt = m
                train_scores, test_scores, _ = model.result()
                writer.collect_results(n, m, train_scores, test_scores)
            
            #running other machine learning methods and printing the results
            ml_model = ml_models(data_formatter, train, test)
            for m in ml_methods:
                ml_model.expt = m
                train_scores, test_scores, _ = ml_model.result()
                writer.collect_results(n, m, train_scores, test_scores)
                                           
        writer.print_results()
        print("Printing average results.....")
        print(writer.average_results)
        
        if "mip_wi" in edcs_methods or "mip" in edcs_methods:
            print("Printing mip performance results.....")
            print(writer.mip_perf)      
          

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
            default="bank_credit",
            choices=datasets,
            help="Dataset Name. Default={}".format(",".join(datasets)))            
        parser.add_argument(
            "edcs",
            metavar="edcs",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="yes",
            help="Whether to use our EDCS algorithms") 
        parser.add_argument(
            "ml",
            metavar="ml",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="no",
            help="Whether to use ML algorithms")        
        parser.add_argument(
            "cost_cla",
            metavar="cost_cla",
            type=str,
            nargs="?",
            choices=["yes", "no"],            
            default="no",
            help="Whether to use CostCla algorithms")                            
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
            default=10,
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
                args.edcs,
                args.ml,
                args.cost_cla,
                args.hyperparam_opt,
                args.time_limit,
                folder
               )

    expt_name, edcs, ml, cost_cla, hyperparam_opt, time_limit, folder = get_args()
    
    config = ExperimentConfig(expt_name, folder)
    formatter = config.make_data_formatter()
           
    formatter.expt_name = expt_name        
    formatter.data_folder = config.data_folder
    formatter.model_folder = config.model_folder
    formatter.results_folder = config.results_folder
    formatter.time_limit = time_limit
    formatter.bayes = (True if hyperparam_opt == "yes" else False)
    use_edcs = (True if edcs == "yes" else False)
    use_ml = (True if ml == "yes" else False)
    use_cost_cla = (True if cost_cla == "yes" else False)

    edcs_methods = []
    ml_methods = []
    
    if (use_edcs):  
        edcs_methods.append("cslr")
        edcs_methods.append("mip")
        edcs_methods.append("mip_wi")
    
    if (use_ml):
        ml_methods.append("logistic")
        ml_methods.append("mip_logistic")
        ml_methods.append("tree")       
        ml_methods.append("xgboost")
        ml_methods.append("svm")
        ml_methods.append("mip_svm")    
        ml_methods.append("svm_cost")    
        ml_methods.append("mip_svm_cost")    
        
    if (use_cost_cla):
        costcla_methods = ["tree_costcla", "rf_costcla", "rp_costcla", "pasting_costcla", "bagging_costcla"]
        ml_methods.append(costcla_methods)
       
    # For new experiments customise inputs to main() .
    main(
        expt_name=expt_name,
        edcs_methods=edcs_methods,
        ml_methods=ml_methods,
        data_formatter=formatter
    ) 

