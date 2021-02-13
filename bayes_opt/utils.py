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

import pandas as pd
import numpy as np

def test_learning(probs, returns):
    """Function to calculate total returns
        
    Args:
        probs, returns: probabilities, returns
        
    Returns:
        gain, outcome: total returns, classes with the greatest returns for each instance
    """
    
    outcome = np.argmax(probs, 1)
    gain = sum(returns[np.arange(len(returns)), np.argmax(probs,1)])
    
    return gain, outcome

class write_results:
    """A helper class to simulate data for Cost Sensitive Learning"
        
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
              'Accuracy': test_scores[1],
              'F1_Score': test_scores[2],                              
             }
        df = pd.DataFrame(data=df, index=[self.index]) 
        self.test_df = pd.concat([self.test_df,df])  
            
        df = {'Repeat_Num': repeat_num,
              'Method': method,
              'Normalized_Return': train_scores[0],
              'Accuracy': train_scores[1],
              'F1_Score': train_scores[2],                              
             }
        df = pd.DataFrame(data=df, index=[self.index]) 
        self.train_df = pd.concat([self.train_df,df])              
            
        if val_scores is not None and val_scores != []:
            
            self.validation = True
            df = {'Repeat_Num': repeat_num,
                  'Method': method,
                  'Normalized_Return': val_scores[0],
                  'Accuracy': val_scores[1],
                  'F1_Score': val_scores[2],                              
                 }
            df = pd.DataFrame(data=df, index=[self.index]) 
            self.val_df = pd.concat([self.val_df,df])

        if method == "mip_wi" or method == "mip":
            
            mip_soln = train_scores[3]
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
        average_train = pd.DataFrame()
        average_test = pd.DataFrame()
        
        for m in self.methods:
            
            df_train = {'Method': m,
                        'Normalized_Return': (self.train_df[self.train_df["Method"] == m].mean()["Normalized_Return"]),
                        'Accuracy': self.train_df[self.train_df["Method"] == m].mean()["Accuracy"],
                        'F1_Score': self.train_df[self.train_df["Method"] == m].mean()["F1_Score"]
                       }
            df_train = pd.DataFrame(data=df_train, index=["Train"])
            average_train = pd.concat([average_train,df_train])
    
            if m == "mip_wi" or m == "mip":
                
                df_opt = {
                'Method': m,
                'F_Mip_Gap': self.mip_perf[self.mip_perf["Method"] == m].mean()["F_Mip_Gap"],
                'F_Time_Passed': self.mip_perf[self.mip_perf["Method"] == m].mean()["F_Time_Passed"],
                'F_Obj_Val': self.mip_perf[self.mip_perf["Method"] == m].mean()["F_Obj_Val"],
                'F_Norm_Obj_Val': self.mip_perf[self.mip_perf["Method"] == m].mean()["F_Norm_Obj_Val"],
                'L_Mip_Gap': self.mip_perf[self.mip_perf["Method"] == m].mean()["L_Mip_Gap"],
                'L_Time_Passed': self.mip_perf[self.mip_perf["Method"] == m].mean()["L_Time_Passed"],
                'L_Obj_Val': self.mip_perf[self.mip_perf["Method"] == m].mean()["F_Obj_Val"],
                'L_Norm_Obj_Val': self.mip_perf[self.mip_perf["Method"] == m].mean()["L_Norm_Obj_Val"],
                }
                df_opt = pd.DataFrame(data=df_opt, index=["MIP"])
                self.mip_avg_perf = pd.concat([self.mip_avg_perf,df_opt])
                self.mip_avg_perf.to_csv(self.results_path+"_mip")
        
        for m in self.methods:
                 
            df_test = {'Method': m,
                       'Normalized_Return': self.test_df[self.test_df["Method"] == m].mean()["Normalized_Return"],
                       'Accuracy': self.test_df[self.test_df["Method"] == m].mean()["Accuracy"],
                       'F1_Score': self.test_df[self.test_df["Method"] == m].mean()["F1_Score"]
                      }
            df_test = pd.DataFrame(data=df_test, index=["Test"]) 
            average_test = pd.concat([average_test,df_test])

        self.average_results = pd.concat([average_train,average_test])  
                        
        if self.validation == True:
            
            average_val = pd.DataFrame()
            self.val_df.to_csv(self.results_path+"_val")            
               
            for m in self.methods:

                df_val = {'Method': m,
                          'Normalized_Return': self.val_df[self.val_df["Method"] == m].mean()["Normalized_Return"],
                          'Accuracy': self.val_df[self.val_df["Method"] == m].mean()["Accuracy"],
                          'F1_Score': self.val_df[self.val_df["Method"] == m].mean()["F1_Score"]
                         }
                df_val = pd.DataFrame(data=df_val, index=["Validation"])  
                average_val = pd.concat([average_val,df_val])  
               
            self.average_results = pd.concat([self.average_results,average_val])  

        self.average_results.to_csv(self.results_path+"_avg")
