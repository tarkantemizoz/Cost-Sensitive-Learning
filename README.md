# A Mixed-Integer Programming Approach To Example-dependent Cost-sensitive Learning

This research is conducted as the Master's Thesis of Tarkan Temizoz.
Query the authors for the full results of this work. To reproduce them, please follow the instructions below.

## Abstract

> In this research, we study example-dependent cost-sensitive learning that brings about varying costs/returns based on the labeling decisions. Originating from decision-making models, these problems are distinguished in areas where cost/return information in data is focal, instead of the true labels. For example, in churn prediction and credit scoring, the primary aim is to build predictive models that minimize misclassification error. Then, the outputs of the model are used to make decisions to minimize/maximize the costs/returns. In other words, prediction and decision making are considered as two separate tasks which may provide local optimal solutions. To resolve such problems, we propose a general strategy to incorporate instance-based costs/returns in a learning algorithm. Specifically, the learning problem is formulated as a mixed-integer program to maximize the total return. Given the high computational complexity of the mixed-integer linear programming problems, this model can be practically inefficient for training on large-scale data sets. To address this, we also propose Cost-sensitive Logistic Regression, a nonlinear approximation of the formulated linear model, which benefits from gradient descent based optimization. Our experimental results show that the proposed approaches provide better total returns compared to traditional learning approaches. Moreover, we show that the optimization performance of the mixed-integer programming solver can be enhanced by providing initial solutions from Cost-sensitive Logistic Regression to the mixed-integer programming model.

## Code
This repository contains the source code of the proposed models for Example-dependent Cost-sensitive learning (EDCS) (see **cslr.py**). These models are: 

* **Cost-sensitive Logistic Regression**: Nonlinear approximation to MIP formulation by optimizing its task using gradient descent algorithm.
* **MIP**: Mixed-integer programming model that maximizes the total return on a training set for EDCS learning problems.
* **MIP-WI**: Mixed-integer programming model with initial solution, which is taken from CSLR and fed into MIP model.

Important modules are defined as following:

* **Models**: Contains the scripts of the proposed methods and other machine learning appraoaches.
* **utils**: Contains the scripts for hyperparameter tuning with Bayesian Optimization and a helper class to write the results.
* **data\_formatters**: Holds the data set specific experiment formats, such as functions for normalization, model parameters etc. Also stores the parent class (see **base.py**) every experiment inherits from.
* **datasets**: Stores the data sets, the features and the returns.
* **expt\_settings**: Holds the folder paths and specifies the experiment to work on.

To reproduce the results:

Set up your directory to project directory, and call:

```
conda env create -f environment.yml
```
Please manually obtain and provide your licence key for Gurobi.

* **train.py**: Runs the experiments with specified models.

## How to Run Default Experiments:
Our default experiments consists of credit scoring ``bank_credit`` problem and four synthetic setups ``ex1``, ``ex2``, ``ex3`` and ``ex4``.
In the research, only ``ex2`` and ``ex4`` experiments are used as synthetic data sets.

To train the models with default parameters, run:

```
python -m train expt_name edcs ml cost_cla hyperparam_opt time_limit output_folder 
```

``expt_name`` denotes the aforementioned default experiments.
``edcs``, ``ml`` and ``cost_cla`` show whether to run the particular models on the specified ``expt_name``, (options are {``yes`` or``no``}).
``hyperparam_opt`` shows whether to use Bayesian Optimization for hyperparameter tuning, (options is {``yes`` or``no``}).
``time_limit`` is the time limit in seconds for ``mip`` and ``mip_wi``.
``output_folder`` is the root folder in which experiment is saved. 


## How to Customize Scripts for new Data Sets

To implement our proposals to new data sets, it requires to create a new formatter and config updates. 
Say the name of the new data set: ``new_example``

### Step 1: Create a new data formatter
Create a new python file new_example.py in ``data_formatters``. It should contain a formatter class (e.g. ``new_example``), which inherits ``base.GenericDataFormatter`` and provides implementations of all abstract functions. 

### Step 2: Update configs.py
Add a name for your new experiement to the ``default_experiments`` attribute in ``expt_settings.configs.ExperimentConfig`` (e.g. ``example``).
```python
default_experiments = ['bank_credit', 'creditcard', 'betting', 'ex1', 'ex2', 'ex3', 'ex4']
```

Add the root of the new formatter in the make_data_formatter function:
```python
def make_data_formatter(self):
    """Gets a data formatter object for experiment.
            
    Returns:
      specified dataformatter.
    """
    dataset = {       
        'bank_credit': data_formatters.bank.bank_credit,
        'new_example': data_formatters.new_example.new_example  # define your new example here.
    }
    for ex in ExperimentConfig.simulated_experiments:
        dataset[ex] = data_formatters.simulation.data_generator
    return dataset[self.experiment]()
```

### Step 3: Run the new experiment 

Call train.py with the new data set.
```bash
python3 -m train new_example
```

## How to Customize Scripts for new synthetic Data Sets

To create a new synthetic data set, e.g. 'ex5', it requires some modifications in simulation.py and config updates. 

### Step 1: Update simulation.py

Add baseline and effect functions for the experiment 'ex5' in decision function:
```python
def decision(self, arr):        
    """Calculates the outcomes based on the baseline and effect functions
            
    Args:
        arr: features
            
    Returns:
        outcomes
    """
    .
    .
    .
    elif self.expt_name == "ex5":

        for i in range(len(arr)):

                y[i,0] = sum(arr[i,(0,1,6,9,14,26,33)])  + abs(arr[i,2])
                .
                .
    .
    .
    .
    return y  
```

Specify the returns for the new experiment by customizing ``returns`` function:
```python
def returns(self, y):
    """Generate return values and labels
            
    Args:
        y: outcomes to determine which class is to be assigned the greatest return
            
    Returns:
        returns and labels
    """
        
    elif self.expt_name == "ex5":
            
        np.random.seed(self.seed)
        ret = np.concatenate(.
                             .
                             .)
        for i in range(len(y)):
            
            returns[i,0] = (ret[i,0] if outcomes[i] == 0 else ret[i,1])
            .
            .
    .
    .
    .                                                
    return returns, outcomes.astype(int)  
```

Lastly, provide the experiment specific parameters.
```python
def simulation_params(self):
    """Returns experiment specific parameters."""
    .   
    .
    .
    params_simul['ex5'] = {
        'num_class': 3,
        'num_features': 30,
        'noise': 0.5,
        'n': 3000,
        'n_test': 20000
    }             
                                
    if params_simul[self.expt_name] is None:
            raise ValueError('Unknown experiment has been chosen! Set your experiment parameters!') 
                
    return params_simul  
```

### Step 2: Update configs.py
Add the name of the new experiement 'ex5' to the ``default_experiments`` and ``simulated_experiments`` attributes in ``expt_settings.configs.ExperimentConfig``.
```python
simulated_experiments = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5]
default_experiments = ['bank_credit', 'creditcard', 'betting', 'ex1', 'ex2', 'ex3', 'ex4']
```

### Step 3: Run the new experiment 

Call train.py with the new synthetic data set.
```
python -m train ex5
```
