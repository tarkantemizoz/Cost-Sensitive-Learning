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
        self.validation = False

    def collect_results(self, 
                        repeat_num,
                        method,
                        train_scores, 
                        test_scores,
                        val_scores=None,
                       ):   
            
    
        df = {'Repeat_Num': repeat_num,
              'Method': method,
              'Normalized_Return': test_scores[0],
              'Accuracy': test_scores[1],
              'F1_Score': test_scores[2],                              
             }
        df = pd.DataFrame(data=df, index=[0]) 
        self.test_df = pd.concat([self.test_df,df])  
            
        df = {'Repeat_Num': repeat_num,
              'Method': method,
              'Normalized_Return': train_scores[0],
              'Accuracy': train_scores[1],
              'F1_Score': train_scores[2],                              
             }
        df = pd.DataFrame(data=df, index=[0]) 
        self.train_df = pd.concat([self.train_df,df])              
            
        if val_scores is not None and val_scores != []:
            
            self.validation = True
            df = {'Repeat_Num': repeat_num,
                  'Method': method,
                  'Normalized_Return': val_scores[0],
                  'Accuracy': val_scores[1],
                  'F1_Score': val_scores[2],                              
                 }
            df = pd.DataFrame(data=df, index=[0]) 
            self.val_df = pd.concat([self.val_df,df])   
                
    def print_results(self):
        
        self.results_path = self.formatter.results_path
        self.test_df.to_csv(self.results_path+"_test")
        self.train_df.to_csv(self.results_path+"_train")

        if self.validation:

            self.val_df.to_csv(self.results_path+"_val")

        
        
        
        
        
        