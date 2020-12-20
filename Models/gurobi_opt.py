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

import numpy as np
from gurobipy import *


class Optimization_MIP:
    
    def __init__(self, config, data, returns):
        
        self.n = config.get("n", data.shape[0])
        self.num_class = returns.shape[1]
        self.num_features = data.shape[1]
        self.beta_lower = config.get("beta_lower", GRB.INFINITY)
        self.score_upper_bound = config.get("upper_bound", 100)
        self.score_lower_bound = config.get("lower_bound", -100)
        self.margin = config.get("margin", 0.01)
        
        self.m = Model("Opt")
        self.beta = self.m.addVars(self.num_class, self.num_features,
                                   vtype=GRB.CONTINUOUS,
                                   name="beta",lb=-self.beta_lower)
        self.x = self.m.addVars(self.n, self.num_class,
              obj=returns,
              vtype=GRB.BINARY,
              name="x")    
        self.p = self.m.addVars(self.n, self.num_class,
              vtype=GRB.CONTINUOUS,
              name="probs",lb=self.score_lower_bound,ub=self.score_upper_bound)
        self.q = self.m.addVars(self.n, self.num_class, self.num_class,
              vtype=GRB.BINARY,
              name="q")        
    
        for i in range(self.n):
            self.m.addConstr(sum(self.x[i,j] for j in range(self.num_class)) == 1)
            for j in range(self.num_class):
                self.m.addConstr((self.p[i,j] - sum(data[i,s] * self.beta[j,s]
                                                    for s in range(self.num_features)) == 0))
                self.m.addConstr(sum(self.q[i,j,t] for t in range((j+1),self.num_class)) -
                                 sum(self.q[i,t,j] for t in range(0,j)) -
                                 self.x[i,j] <= self.num_class - (j + 1) - 1)   
                
            for j in range((self.num_class - 1)):
                
                for t in range((j + 1), self.num_class):
                    self.m.addConstr(self.p[i,j] - self.p[i,t] -
                                     (self.score_upper_bound - self.score_lower_bound) * self.q[i,j,t] <= -self.margin)
                    self.m.addConstr(self.p[i,j] - self.p[i,t] +
                                     (self.score_upper_bound - self.score_lower_bound) * (1 - self.q[i,j,t]) >= self.margin)

