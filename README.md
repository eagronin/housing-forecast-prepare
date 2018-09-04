# Data Preparation and Cleaning

## Overview
This section first describes cleaning and processing of the data on single-family homes in Denver, CO.  It then turns to construction of new features that were subsequently used for predicting home values along with the existing features.  The new features include comparables-based valuation and forecaseted value using the history of prior sales.  The existing features were listed in the [previous section](https://eagronin.github.io/housing-forecast-acquire/).  Home values are subsequently predicted using a random forest model fitted to a training set and evaluated using a test set.  The model produced the test set R-squared of 0.92.  The subsequent sections discuss the steps that I took in more detail. 

Description of the data is provided in the [previous section](https://eagronin.github.io/housing-forecast-acquire/).

Estimation of the model and evaluation of its performance are discussed in the [next section](https://eagronin.github.io/housing-forecast-analyze/).

The analysis for this project was performed in Python.

## Data Cleaning and Processing 

