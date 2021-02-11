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
    """ A helper class to create mixed-integer programming model EDCS learning problem
        
    Attributes:
        n: number of instances in the data set
        num_class: number of classes of an instance
        num_features: number of features of an instance
        beta_lower: lower bound on coefficients
        score_upper_bound: upper bound on the scores
        score_lower_bound: lower bound on the scores
        margin: minimum margin between score values of an instance
        m: the model
        beta: coefficients
        d: binary variable showing whether the instance i is predicted as class j
        p: scores
        q: binary variable showing whether the score of class j is greater
            than the score of class t
    """
    
    def __init__(self, config, data, returns):
        """Initialized the model and build the objective function and the constraints
            
        Args:
            config: configuration for the experiment
            data: the data set
            returns: returns of the data
        """
        
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
        self.d = self.m.addVars(self.n, self.num_class,
              obj=returns,
              vtype=GRB.BINARY,
              name="d")
        self.p = self.m.addVars(self.n, self.num_class,
              vtype=GRB.CONTINUOUS,
              name="probs",lb=self.score_lower_bound,ub=self.score_upper_bound)
        self.q = self.m.addVars(self.n, self.num_class, self.num_class,
              vtype=GRB.BINARY,
              name="q")        
    
        for i in range(self.n):
            
            # ensurinh each instance is labeled as only one class
            self.m.addConstr(sum(self.d[i,j] for j in range(self.num_class)) == 1)
            
            for j in range(self.num_class):
                
                # determining the score values
                self.m.addConstr((self.p[i,j] - sum(data[i,s] * self.beta[j,s]
                                                    for s in range(self.num_features)) == 0))
                                                    
                # determining the labelling decisions
                self.m.addConstr(sum(self.q[i,j,t] for t in range((j+1),self.num_class)) -
                                 sum(self.q[i,t,j] for t in range(0,j)) -
                                 self.d[i,j] <= self.num_class - (j + 1) - 1)
                
            for j in range((self.num_class - 1)):
                
                for t in range((j + 1), self.num_class):
                    
                    # conducting pairwise comparisons
                    self.m.addConstr(self.p[i,j] - self.p[i,t] -
                                     (self.score_upper_bound - self.score_lower_bound) * self.q[i,j,t] <= -self.margin)
                    self.m.addConstr(self.p[i,j] - self.p[i,t] +
                                     (self.score_upper_bound - self.score_lower_bound) * (1 - self.q[i,j,t]) >= self.margin)

