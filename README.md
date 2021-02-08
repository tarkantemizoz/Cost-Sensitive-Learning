# Cost-Sensitive-Learning 
A Mixed-Integer Programming Approach To Example-dependent Cost-sensitive Learning

Authors: Tarkan Temizoz, Mustafa G. Baydogan, Mert Yuksekgonul

This research is conducted as the Master's Thesis of Tarkan Temizoz.
Query the authors to request the results of this work. To reproduce them, please follow the instructions below.

## Abstract

In this research, we study example-dependent cost-sensitive learning that brings about varying costs/returns based on the labeling decisions. Originating from decision-making models, these problems are distinguished in areas where cost/return information in data is focal, instead of the true labels. For example, in churn prediction and credit scoring, the primary aim is to build predictive models and decision rules to maximize/minimize the returns/costs of the companies. Traditional accuracy-driven classification methods do not take instance-based costs/returns into account; instead, learning is performed based on constant misclassification error. Therefore, we propose a general strategy to incorporate instance-based costs/returns in a learning algorithm. Specifically, the learning problem is formulated as a mixed-integer program to maximize the total return. Given the high computational complexity of the mixed-integer linear programming problems, this model can be practically inefficient for training on large-scale data sets. To address this, we also propose Cost-sensitive Logistic Regression, a nonlinear approximation of the formulated linear model, which benefits from gradient descent based optimization by using deep learning tools. Our experimental results show that the proposed approaches provide better total returns compared to traditional learning approaches. Moreover, we show that by providing initial solutions from Cost-sensitive Logistic Regression to the linear programming model, the optimization performance of the mixed-integer programming solver can be enhanced.

## How to Run Default Experiments

This repository contains the source code of the proposed models for Example-dependent Cost-sensitive learning (EDCS) (see **cslr.py**). These models are: 

* **Cost-sensitive Logistic Regression**: Nonlinear approximation to MIP formulation by optimizing its task using gradient descent algorithm.
* **MIP**: Mixed-integer programming model that maximizes the total return on a training set for EDCS learning problems.
* **MIP-WI**: Mixed-integer programming model with initial solution, which is taken from CSLR and fed into MIP model.


```bash
py -m train expt_name cslr mip_wi mip ml hyperparam_opt time_limit output_folder
```
(options are {'yes' or'no'}).
Specify the name of the experiment expt_name and choose which method(s) to run.
To run all models  full , 


## How to Customize for new Data Sets

### Step 1: Create custom data file

First, create inheriting 
First, create a new python file in ``data_formatters`` (e.g. example.py) which contains a data formatter class (e.g. ``ExampleFormatter``). This should inherit ``base.GenericDataFormatter`` and provide implementations of all abstract functions. An implementation example can be found in volatility.py.

If a 

### Step 1:
```python
    def make_data_formatter(self):
        """Gets a data formatter object for experiment.
        Returns:
          Default DataFormatter per experiment.
        """
        dataset = {       
            'bank_credit': data_formatters.bank.bank_credit
            'example': data_formatters.example.ExampleFormatter, # new data set here!
        }
        for ex in ExperimentConfig.simulated_experiments:
            dataset[ex] = data_formatters.simulation.data_generator
        
        return dataset[self.experiment]()
```
If the new data set is created

### Step 2:


### Step 3: Run train.py

To run churn data set without hyperparameter optimization

```bash
py -m train churn yes yes yes yes no 300 
```


```python
default_experiments = ['volatility', 'electricity', 'traffic', 'favorita', 'example']
```

        params_simul['ex4'] = {
            'num_class': 3,
            'num_features': 25,
            'noise': 1,
            'n': 1000,
            'n_test': 20000
        } 


