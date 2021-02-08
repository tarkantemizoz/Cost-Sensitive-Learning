# Cost-Sensitive-Learning 
A Mixed-Integer Programming Approach To Example-dependent Cost-sensitive Learning

## Abstract

In this research, we study example-dependent cost-sensitive learning that brings about varying costs/returns based on the labeling decisions. Originating from decision-making models, these problems are distinguished in areas where cost/return information in data is focal, instead of the true labels. For example, in churn prediction and credit scoring, the primary aim is to build predictive models and decision rules to maximize/minimize the returns/costs of the companies. Traditional accuracy-driven classification methods do not take instance-based costs/returns into account; instead, learning is performed based on constant misclassification error. Therefore, we propose a general strategy to incorporate instance-based costs/returns in a learning algorithm. Specifically, the learning problem is formulated as a mixed-integer program to maximize the total return. Given the high computational complexity of the mixed-integer linear programming problems, this model can be practically inefficient for training on large-scale data sets. To address this, we also propose Cost-sensitive Logistic Regression, a nonlinear approximation of the formulated linear model, which benefits from gradient descent based optimization by using deep learning tools. Our experimental results show that the proposed approaches provide better total returns compared to traditional learning approaches. Moreover, we show that by providing initial solutions from Cost-sensitive Logistic Regression to the linear programming model, the optimization performance of the mixed-integer programming solver can be enhanced.

# How to run default experiments


```bash
py -m train expt_name cslr mip_wi mip ml hyperparam_opt time_limit output_folder
```
(options are {'yes' or'no'}).
Specify the name of the experiment expt_name and choose which method(s) to run.
For full , 


# Customizing for new data sets
