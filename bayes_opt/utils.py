#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

def test_learning(probs, returns):   
    gain = np.zeros(len(probs))
    outcome = np.zeros(len(probs))
    for i in range(len(probs)):            
        gain[i] = returns[i,np.argmax(probs[i])]
        outcome[i] = np.argmax(probs[i])
    return sum(gain), outcome

class write_results:
    
    def __init__(self, formatter):
        
        self.formatter = formatter
        self.test_df = pd.DataFrame()
        self.train_df = pd.DataFrame()        
        self.val_df = pd.DataFrame()
        self.average_results = pd.DataFrame()
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
            
        self.index += 1         
        
    def print_results(self):
        
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

