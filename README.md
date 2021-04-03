# ML-DL-implementation
[![Build Status](https://travis-ci.org/RoboticsClubIITJ/ML-DL-implementation.svg?branch=master)](https://travis-ci.org/RoboticsClubIITJ/ML-DL-implementation)
[![Gitter](https://badges.gitter.im/ML-DL-implementation/community.svg)](https://gitter.im/ML-DL-implementation/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/RoboticsClubIITJ/ML-DL-implementation)
<a href="https://devlup-labs.github.io"><img src="https://img.shields.io/badge/Maintained%20under-Winter%20of%20Code%2C%20DevlUp%20Labs-brightgreen"/></a>

Machine Learning and Deep Learning library in python using numpy and matplotlib.

## Why this repository?
-----------------------

This repository gives beginners and newcomers in
the field of AI and ML a chance to understand the
inner workings of popular learning algorithms by presenting them with a simple to analyze the implementation of ML and DL algorithms in pure python using only numpy as a backend for linear algebraic computations for the sake of efficiency.

The goal of this repository is not to create the most efficient implementation but the most transparent one, so that anyone with little knowledge of the field can contribute and learn.

Installation
------------

You can install the library by running the following command,

```python
python3 setup.py install
```

For development purposes, you can use the option `develop` as shown below,

```python
python3 setup.py develop
```
   
Testing
-------

For testing your patch locally follow the steps given below,

1. Install [pytest-cov](https://pypi.org/project/pytest-cov/). Skip this step if you are already having the package.
2. Run, `python3 -m pytest --doctest-modules --cov=./ --cov-report=html`. Look for, `htmlcov/index.html` and open it in your browser, which will show the coverage report. Try to ensure that the coverage is not decreasing by more than 1% for your patch.


## Contributing to the repository

Follow the following steps to get started with contributing to the repository.

- Clone the project to you local environment.
  Use
  `git clone https://github.com/RoboticsClubIITJ/ML-DL-implementation/`
  to get a local copy of the source code in your environment.

- Install dependencies: You can use pip to install the dependendies on your computer.
  To install use
  `pip install -r requirements.txt`

- Installation:
  use `python setup.py develop` if you want to setup for development or `python setup.py install` if you only want to try and test out the repository.

- Make changes, work on a existing issue or create one. Once assigned you can start working on the issue.

- While you are working please make sure you follow standard programming guidelines. When you send us a PR, your code will be checked for PEP8 formatting and soon some tests will be added so that your code does not break already existing code. Use tools like flake8 to check your code for correct formatting.


# Algorithms Implemented

| Activations | Location |  Optimizers | Location | Models | Location | Backend | Location | Utils | Location |
| :------------ | ------------: | :------------ | ------------: | :------------ | ------------: | ------------: | ------------: | ------------: | -----------: |
| **ACTIVATION FUNCTIONS**| |**OPTIMIZERS**| | **MODELS** | | **BACKEND** | | **PREPROCESSOR UTILS** |
| Sigmoid | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L4) | Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L6) | Linear Regression | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L31) | Autograd | [autograd.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/autograd.py) | Bell Curve | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#29)  
| Tanh | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L46) | Stochastic Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L59) | Logistic Regression| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L578) | Tensor | [tensor.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/tensor.py)| Standard_Scaler | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#53)
| Softmax | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L86) | Mini Batch Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L126) | Decision Tree Classifier| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L775)| Functions | [functional.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/functional.py) | MaxAbs_Scaler | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#53) |
| Softsign | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L134) | Momentum Gradient Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L204) | KNN Classifier/Regessor| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1034) | | | Z_Score_Normalization | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#L117) |
| Relu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L174) | Nesterov Accelerated Descent | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L297) | Naive Bayes | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1119)| | | Mean_Normalization | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#L139) |
| Leaky Relu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L214) | Adagrad | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L392) | Gaussian Naive Bayes| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1179) | | | Min Max Normalization | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#L139) |
| Elu | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L261) | Adadelta | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L467) | Multinomial Naive Bayes | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1257) | | | Feature Clipping | [preprocessor_utils.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/utils/preprocessor_utils.py#L94) |
| Swish | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L305) | Adam | [optimizers.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/optimizers.py#L545) | Polynomial Regression | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L268) |
| Unit Step | [activations.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/activations.py#L285) | | | Bernoli Naive Bayes | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1233) |
| | | | | Random Forest Classifier | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L931) | 
| | | | | K Means Clustering| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1279) |
| | | | | Divisive Clustering | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1370) |
| | | | | Agglomerative Clustering | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1717) |
| | | | | BayesOptimization | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1443) |
| | | | | Numerical Outliers| [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1598) |
| | | | | Principle Component Analysis | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1489) |
| | | | | Z_Score | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1637) |
| | | | | Sequential Neural Network | [models.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/models.py#L1680) |

| Loss Functions | Location | Regularizer | Location | Metrics | Location |
| :------------ | ------------: | :------------ | ------------: | :------------ | ------------: |
|**LOSS FUNCTIONS**| |**REGULARIZER**| |**METRICS**| | 
| Mean Squared Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L5) | L1_Regularizer| [regularizer.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/regularizer.py#L9) | Confusion Matrix | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L25) 
| Logarithmic Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L57) | L2_Regularizer | [regularizer.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/regularizer.py#L58) | Precision | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L81) 
| Absolute Error | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L113) | LinearRegWith_Regularization | [regularizer.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/regularizer.py#L106) | Accuracy | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L80) 
| Cosine Similarity | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L173) | | | Recall | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L82) 
| Log_cosh | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L248) | | | F1 Score | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L85) 
| Huber | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L300) | | | FbTheta | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L88) 
| MeanSquaredLogLoss | [loss_func.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/loss_func.py#L367) | | | Specificity | [metrics.py](https://github.com/RoboticsClubIITJ/ML-DL-implementation/blob/master/MLlib/metrics.py#L86) 

