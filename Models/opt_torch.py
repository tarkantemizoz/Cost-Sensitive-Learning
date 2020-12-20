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

import time
import torch
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from Models.linearnet import LinearNet

class Optimization:
    """ A helper class to train, test and diagnose the Cost Sensitive Learning"""

    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.train_return = []
        self.val_return = []
        self.validation = False
        self.batch_size = config.get("batch_size",32)
        self.n_epochs = config.get("n_epochs", 1000)
        self.n_steps = config.get("n_steps", 500)     
    
    @staticmethod
    def batch(iterable, n):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]        
            
    def train(self, x_train, r_train, x_val=None, r_val=None):        
        if x_val is not None or r_val is not None:
            self.validation = True
        start_time = time.time()
                    
        for epoch in range(self.n_epochs):
            x_shuff, r_shuff = shuffle(x_train, r_train)
            self.model.train()            
            for j in self.batch(range(0, len(x_shuff)),self.batch_size):
                if len(j) < 2:
                    break
                x_batch = x_shuff[j]
                r_batch = r_shuff[j]
                self.optimizer.zero_grad()
                outputs, _, _ = self.model(x_batch)
                loss = -torch.mul(outputs, r_batch).sum()                              
                loss.backward()
                self.optimizer.step()
                            
            returns_train, _, _ = self.evaluate(x_train, r_train)
            self.train_return.append(returns_train)
            if self.validation is True:
                returns_val, _, _ = self.evaluate(x_val, r_val)
                self.val_return.append(returns_val)
                          
            if ((epoch+1) % self.n_steps == 0):
                elapsed = time.time() - start_time
                print(
                    ("Epoch %d Train Return: %.3f.")  % (epoch + 1, self.train_return[-1]),
                    ((" Validation Return: %.3f. Elapsed time: %.3fs.")
                     % (self.val_return[-1], elapsed)
                     if self.validation is True else 
                     " Elapsed time: %.3fs."
                     % elapsed) 
                )
                start_time = time.time()        
                
    def evaluate(self, x_test, r_test):
        with torch.no_grad():
            outputs, probs, _ = self.model(x_test)
            returns = torch.mul(outputs, r_test).sum()
            return returns, outputs, probs           
           
    def plot_return(self):
        plt.plot(self.train_return, label="Train Return")
        plt.plot(self.val_return, label="Test Return")
        plt.legend()
        plt.title("Returns")

