import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score


def test_performance(true_labels, predictions, returns, only_returns = False):
    
    min_total_returns = sum(np.amin(returns, 1))
    max_total_returns = sum(np.amax(returns, 1))

    total_returns = sum(returns[np.arange(len(returns)), predictions])
    normalized_returns = (total_returns - min_total_returns) / (max_total_returns - min_total_returns)

    if (only_returns):
        return [normalized_returns, 0, 0, 0]    
    else:        
        savings = (total_returns - np.max(np.sum(returns,0))) / abs(np.max(np.sum(returns,0))) 
        acc, f1 = (accuracy_score(true_labels, predictions), f1_score(true_labels, predictions, average="weighted"))  

        return [normalized_returns, savings, acc, f1]    

    
class write_results:
    """A helper class for collecting and writing results for Example Dependent Cost Sensitive Learning experiments"
        
    Attributes:
        formatter: formatter of the specified experiment
        test_df: dataframe to hold test results
        train_df: dataframe to hold train results
        val_df: dataframe to hold validation results
        mip_perf: dataframe to hold mip and mip-wi results seperately
        average_results: dataframe to hold average results
        mip_avg_perf: dataframe to hold average results of mip and mip-wi
        methods: list to store the methods
        validation: whether there are results of the validation folds
    """
    
    def __init__(self, formatter):
            
        self.formatter = formatter
        self.test_df = pd.DataFrame()
        self.train_df = pd.DataFrame()        
        self.val_df = pd.DataFrame()
        self.mip_perf = pd.DataFrame()
        self.average_results = pd.DataFrame()
        self.mip_avg_perf = pd.DataFrame()
        self.methods = []
        self.validation = False
        self.index = 0

    def collect_results(self, 
                        repeat_num,
                        method,
                        train_scores, 
                        test_scores,
                        val_scores=None,
                       ):   
        """Prepare the results to be printed
            
        Args:
            repeat num: number of repeat
            method: name of the model results belong
            train_scores: train results
            test_scores: test results
            val_scores: validation results
        """

        self.methods.append(method)
        
        df = {'Repeat_Num': repeat_num,
              'Method': method,
              'Normalized_Return': test_scores[0],
              'Savings': test_scores[1],
              'Accuracy': test_scores[2],
              'F1_Score': test_scores[3],                              
             }
        df = pd.DataFrame(data=df, index=[self.index]) 
        self.test_df = pd.concat([self.test_df,df])  
            
        df = {'Repeat_Num': repeat_num,
              'Method': method,
              'Normalized_Return': train_scores[0],
              'Savings': train_scores[1],
              'Accuracy': train_scores[2],
              'F1_Score': train_scores[3],                              
             }
        df = pd.DataFrame(data=df, index=[self.index]) 
        self.train_df = pd.concat([self.train_df,df])       
        
        if val_scores is not None and val_scores != []:
            
            self.validation = True
            df = {'Repeat_Num': repeat_num,
                  'Method': method,
                  'Normalized_Return': val_scores[0],
                  'Savings': val_scores[1],
                  'Accuracy': val_scores[2],
                  'F1_Score': val_scores[3],                              
                 }
            df = pd.DataFrame(data=df, index=[self.index]) 
            self.val_df = pd.concat([self.val_df,df])

        if method == "mip_wi" or method == "mip":

            mip_soln = train_scores[4]
            first_feasible = mip_soln["first_feasible"]
            last_feasible = mip_soln["last_feasible"]
            df = {'Method': method,
                  'F_Mip_Gap': first_feasible["mip_gap"]*100,
                  'F_Time_Passed': first_feasible["time_passed"],
                  'F_Obj_Val': first_feasible["obj_val"],
                  'F_Norm_Obj_Val': first_feasible["norm_obj_val"],
                  'L_Mip_Gap': last_feasible["mip_gap"]*100,
                  'L_Time_Passed': last_feasible["time_passed"],
                  'L_Obj_Val': last_feasible["obj_val"],
                  'L_Norm_Obj_Val': last_feasible["norm_obj_val"],
                  }
            df = pd.DataFrame(data=df, index=[self.index])
            self.mip_perf = pd.concat([self.mip_perf,df])

        self.index += 1         
    
    def print_results(self):
        """Prints the results."""

        self.methods = list(set(self.methods))
        self.results_path = self.formatter.results_path
        self.test_df.to_csv(self.results_path+"_test")
        self.train_df.to_csv(self.results_path+"_train")
   
        # Group by 'Method' and calculate the mean
        df_train = self.train_df.groupby('Method').mean()
        df_train = df_train.drop(columns=['Repeat_Num'])
        multi_index = pd.MultiIndex.from_product([['Train'], df_train.index], names=['Set', 'Method'])
        df_train.index = multi_index

        df_test = self.test_df.groupby('Method').mean()
        df_test = df_test.drop(columns=['Repeat_Num'])
        multi_index = pd.MultiIndex.from_product([['Test'], df_test.index], names=['Set', 'Method'])
        df_test.index = multi_index
        
        self.average_results = pd.concat([df_train,df_test])  
   
        if self.validation == True:
            
            self.val_df.to_csv(self.results_path+"_val")            

            df_val = self.val_df.groupby('Method').mean()
            df_val = df_val.drop(columns=['Repeat_Num'])
            multi_index = pd.MultiIndex.from_product([['Val'], df_val.index], names=['Set', 'Method'])
            df_val.index = multi_index
        
            self.average_results = pd.concat([self.average_results,df_val])  

        self.average_results.to_csv(self.results_path+"_avg")

        if "mip_wi" in self.methods or "mip" in self.methods:

            self.mip_avg_perf = self.mip_perf.groupby('Method').mean()
            self.mip_avg_perf.to_csv(self.results_path+"_mip")
        
                
