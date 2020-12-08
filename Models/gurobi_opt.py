#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from gurobipy import *

class Optimization_MIP_2:
    
    def __init__(self, config, data, returns):
        
        self.n = config.get("n", data.shape[0])
        self.num_class = returns.shape[1]
        self.num_features = data.shape[1]
        self.num_relation = self.num_class * (self.num_class - 1)
        self.beta_lower = config.get("beta_lower", GRB.INFINITY)
        self.score_bound = config.get("bound", 100)
        
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
              name="probs",lb=-self.score_bound,ub=self.score_bound)
        self.q = self.m.addVars(self.n, self.num_relation,
              vtype=GRB.BINARY,
              name="q")        
    
        for i in range(self.n):
            self.m.addConstr(sum(self.x[i,k] for k in range(self.num_class)) == 1)
            self.m.addConstr(sum(self.q[i,k] for k in range(self.num_relation)) == (self.num_relation) / 2)
            for j in range(self.num_class):
                self.m.addConstr((self.p[i,j] - sum(data[i,s] * self.beta[j,s] for s in range(self.num_features)) == 0)) 
                self.m.addConstr(sum(self.q[i,(j * (self.num_class - 1) + k)] for k in range(self.num_class - 1))
                                 - self.x[i,j] <= (self.num_class - 2))
                class_set = np.delete(list(range(0,self.num_class)),j)
                s = 0
                for t in class_set:
                    self.m.addConstr(self.p[i,j] - self.p[i,t] - 2 * self.score_bound * self.q[i,(j * (self.num_class - 1) + s)]
                                     <= -1/self.score_bound)  
                    s += 1

class Optimization_MIP_Indicator:
    
    def __init__(self, config, data, returns):
        
        self.n = config.get("n", data.shape[0])
        self.num_class = returns.shape[1]
        self.num_features = data.shape[1]
        self.beta_lower = config.get("beta_lower", GRB.INFINITY)
        self.score_bound = config.get("bound", 100)
        
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
              name="probs",lb=-self.score_bound,ub=self.score_bound)
        self.q = self.m.addVars(self.n,
              vtype=GRB.BINARY,
              name="q")        
    
        for i in range(self.n):
            self.m.addConstr(sum(self.x[i,j] for j in range(self.num_class)) == 1)
            
            for j in range(self.num_class):
                self.m.addConstr((self.p[i,j] - sum(data[i,k] * self.beta[j,k] for k in range(self.num_features)) == 0)) 
                self.m.addConstr(self.q[i] - self.p[i,j] >= 0) 
                self.m.addGenConstrIndicator(self.x[i,j], True, self.q[i] - self.p[i,j] <= 0)
                class_set = np.delete(list(range(0,self.num_class)),j)
                for t in class_set:
                    self.m.addConstr(self.p[i,j] - self.p[i,t] - 2 * self.score_bound * self.x[i,j]
                                     <= -1/self.score_bound)  
               
                

class Optimization_MIP_2:
    
    def __init__(self, config, data, returns):
        
        self.n = config.get("n", data.shape[0])
        self.num_class = returns.shape[1]
        self.num_features = data.shape[1]
        self.beta_lower = config.get("beta_lower", GRB.INFINITY)
        self.score_upper_bound = config.get("upper_bound", 100)
        self.score_lower_bound = config.get("lower_bound", -100)
        self.margin = config.get("margin", 0.1)
        
        self.m = Model("Opt")
        self.beta = self.m.addVars(self.num_class, self.num_features,
              vtype=GRB.CONTINUOUS,
              name="beta",lb=-self.beta_lower)   
        self.P = self.m.addVars(self.n, self.num_class, self.num_class,
                               vtype=GRB.BINARY,
                               name="P")        
        self.scores = self.m.addVars(self.n, self.num_class,
              vtype=GRB.CONTINUOUS,
              name="scores",lb=self.score_lower_bound,ub=self.score_upper_bound)    
        self.q = self.m.addVars(self.n, self.num_class, self.num_class,
              vtype=GRB.CONTINUOUS,
              name="q")        

        self.y = self.m.addVars(self.n, self.num_class,
              vtype=GRB.CONTINUOUS,
              name="y")       
        
        self.m.setObjective(sum(sum(returns[i,j] * self.P[i,0,j] for j in range(self.num_class))
                                for i in range(self.n)), GRB.MAXIMIZE)        
        for i in range(self.n):
            for j in range(self.num_class):
                self.m.addConstr((self.scores[i,j] -
                                  sum(data[i,k] * self.beta[j,k] for k in range(self.num_features)) == 0)) 
                self.m.addConstr(sum(self.P[i,j,t] for t in range(self.num_class)) == 1)
                self.m.addConstr(sum(self.P[i,t,j] for t in range(self.num_class)) == 1)
                for t in range(self.num_class):
                    self.m.addConstr(self.scores[i,j] - (self.score_upper_bound - self.score_lower_bound) *
                                     (1 - self.P[i,t,j]) - self.q[i,t,j] <= 0)
                    self.m.addConstr(self.q[i,t,j] - self.scores[i,j] <= 0)
                    self.m.addConstr(self.q[i,t,j] - self.P[i,t,j] * self.score_upper_bound <= 0)
                    self.m.addConstr(self.q[i,t,j] - self.P[i,t,j] * self.score_lower_bound >= 0)
                self.m.addConstr(sum(self.q[i,j,s] for s in range(self.num_class)) - self.y[i,j] == 0)    
                if j < (self.num_class - 1):
                    self.m.addConstr(self.y[i,j] - self.y[i,j+1] >= self.margin)                    

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
                self.m.addConstr((self.p[i,j] - sum(data[i,s] * self.beta[j,s] for s in range(self.num_features)) == 0))              
                self.m.addConstr(sum(self.q[i,j,t] for t in range((j+1),self.num_class)) -
                                 sum(self.q[i,t,j] for t in range(0,j)) -
                                 self.x[i,j] <= self.num_class - (j + 1) - 1)   
                
                #for t in range(0, (j+1)):                
                 #   self.m.addConstr(self.q[i,j,t] == 0)
                
            for j in range((self.num_class - 1)):
                
                for t in range((j + 1), self.num_class):
                    self.m.addConstr(self.p[i,j] - self.p[i,t] -
                                     (self.score_upper_bound - self.score_lower_bound) * self.q[i,j,t] <= -self.margin)
                    self.m.addConstr(self.p[i,j] - self.p[i,t] +
                                     (self.score_upper_bound - self.score_lower_bound) * (1 - self.q[i,j,t]) >= self.margin)
                                        


                

                    
                    