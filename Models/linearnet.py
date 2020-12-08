#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module): 
    def __init__(self, config):
        super(LinearNet, self).__init__()
        self.input_size = config["n_inputs"]
        self.dnn_layers = config.get("dnn_layers", 1)        
        self.hidden_size = config.get("hidden_size", 0)
        self.n_outputs = config["n_outputs"]
        self.batch_norm = config.get("batch_norm", False)
        self.relu = nn.ReLU()
        self.action_max = nn.Softmax()
        for i in range(self.dnn_layers):
            self.add_module('layer_' + str(i), nn.Linear(
                        in_features=self.input_size if i == 0 else self.hidden_size[i-1],
                        out_features=self.n_outputs if (i+1 == self.dnn_layers) else self.hidden_size[i],
                        bias=False if self.dnn_layers == 1 else True
            ))
            if self.batch_norm == True:
                self.add_module('batch_' + str(i), nn.BatchNorm1d(
                    self.n_outputs if (i+1 == self.dnn_layers) else self.hidden_size[i]
                ))        
    def forward(self, X):       
        logits = getattr(self, 'layer_'+str(0))(X)
        if self.batch_norm == True:
            logits = getattr(self, 'batch_'+str(0))(logits)
        for i in range(self.dnn_layers-1):
            logits = self.relu(logits)
            logits = getattr(self, 'layer_'+str(i+1))(logits)
            if self.batch_norm == True:
                logits = getattr(self, 'batch_'+str(i+1))(logits)
        output = self.action_max((logits)/0.001)
        output_probs = self.action_max((logits))
        return output, output_probs, logits
