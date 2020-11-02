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
        self.batchnorm = config.get("batchnorm", False)
        self.relu = nn.ReLU()
        self.action_max = nn.Softmax()
        for i in range(self.dnn_layers):
            self.add_module('layer_' + str(i), nn.Linear(
                        in_features=self.input_size if i == 0 else self.hidden_size[i-1],
                        out_features=self.n_outputs if (i+1 == self.dnn_layers) else self.hidden_size[i],
                        bias=False if self.dnn_layers == 1 else True
            ))
            if self.batchnorm == True:
                self.add_module('batch_' + str(i), nn.BatchNorm1d(
                    self.n_outputs if (i+1 == self.dnn_layers) else self.hidden_size[i]
                ))        
    def forward(self, X):       
        logits = getattr(self, 'layer_'+str(0))(X)
        if self.batchnorm == True:
            logits = getattr(self, 'batch_'+str(0))(logits)
        for i in range(self.dnn_layers-1):
            logits = self.relu(logits)
            logits = getattr(self, 'layer_'+str(i+1))(logits)
            if self.batchnorm == True:
                logits = getattr(self, 'batch_'+str(i+1))(logits)
        output = self.action_max((logits)/0.001)
        output_probs = self.action_max((logits))
        return output, output_probs, logits

class LinearNet_1(nn.Module): 
    def __init__(self, config):
        super(LinearNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.action_max = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.action_max((x)/0.001)
        output_probs = self.action_max(x)
        return output, output_probs, x