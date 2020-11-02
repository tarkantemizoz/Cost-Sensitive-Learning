#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, cohen_kappa_score
import numpy as np


def give_set(rep, set_num, val_num, val_sets):
    return val_sets[0][rep][set_num][val_num], val_sets[1][rep][set_num][val_num]

def give_score(gain, y_pred, y_true, which_score):
    if which_score == "accuracy":        
        score = accuracy_score(y_true, y_pred)       
    elif which_score == "precision":                
        score = precision_score(y_true, y_pred)    
    elif which_score == "recall":            
        score = recall_score(y_true, y_pred)    
    elif which_score == "f1":             
        score = f1_score(y_true, y_pred)        
    elif which_score == "kappa":                
        score = cohen_kappa_score(y_true, y_pred)         
    else:       
        score = gain        
    return score

def test_learning(probs, returns):   
    gain = np.zeros(len(probs))
    outcome = np.zeros(len(probs))
    for i in range(len(probs)):            
        gain[i] = returns[i,np.argmax(probs[i])]
        outcome[i] = np.argmax(probs[i])
    return sum(gain), outcome

