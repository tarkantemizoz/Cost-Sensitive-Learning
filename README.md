# A Mixed-Integer Programming Approach To Example-dependent Cost-sensitive Learning

Authors: Tarkan Temizoz, Mustafa G. Baydogan, Mert Yuksekgonul

This research is conducted as the Master's Thesis of Tarkan Temizoz.
Query the authors to request the full results of this work. To reproduce them, please follow the instructions below.

## Abstract

In this research, we study example-dependent cost-sensitive learning that brings about varying costs/returns based on the labeling decisions. Originating from decision-making models, these problems are distinguished in areas where cost/return information in data is focal, instead of the true labels. For example, in churn prediction and credit scoring, the primary aim is to build predictive models and decision rules to maximize/minimize the returns/costs of the companies. Traditional accuracy-driven classification methods do not take instance-based costs/returns into account; instead, learning is performed based on constant misclassification error. Therefore, we propose a general strategy to incorporate instance-based costs/returns in a learning algorithm. Specifically, the learning problem is formulated as a mixed-integer program to maximize the total return. Given the high computational complexity of the mixed-integer linear programming problems, this model can be practically inefficient for training on large-scale data sets. To address this, we also propose Cost-sensitive Logistic Regression, a nonlinear approximation of the formulated linear model, which benefits from gradient descent based optimization by using deep learning tools. Our experimental results show that the proposed approaches provide better total returns compared to traditional learning approaches. Moreover, we show that by providing initial solutions from Cost-sensitive Logistic Regression to the linear programming model, the optimization performance of the mixed-integer programming solver can be enhanced.

## How to Run Default Experiments

This repository contains the source code of the proposed models for Example-dependent Cost-sensitive learning (EDCS) (see **cslr.py**). These models are: 

* **Cost-sensitive Logistic Regression**: Nonlinear approximation to MIP formulation by optimizing its task using gradient descent algorithm.
* **MIP**: Mixed-integer programming model that maximizes the total return on a training set for EDCS learning problems.
* **MIP-WI**: Mixed-integer programming model with initial solution, which is taken from CSLR and fed into MIP model.



