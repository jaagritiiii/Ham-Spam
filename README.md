
# Ham-Spam Email Classifier


## Table of Contents
- [Introduction](##introduction)
- [Problem Statement](##problem-statement)
- [Data](##data)
- [Prerequisites](##prerequisites)
- [Installation](##installation)
- [Usage](##usage)
- [Functions Explanation](##functions-explanation)
  - [train_test_split](##train_test_split)
  - [accuracy_score](##accuracy_score)
  - [Prediction](##prediction)
- [Logistic Regression for Binary Classification](##logistic-regression-for-binary-classification)
- [Model Performance](##model-performance)
- [Deployment](##deployment)
- [License](##license)

## Introduction
Email spam is a persistent issue that can clutter your inbox with unwanted and potentially harmful messages. The **Ham-Spam Email Classifier** is an ML model built using logistic regression with Python's Scikit-Learn, NumPy, and Pandas libraries to predict whether an email is "ham" (legitimate) or "spam" with high accuracy. It has been developed in Google Colab, a convenient and collaborative environment for data preprocessing, modeling, and evaluation.

## Problem Statement
Unwanted spam emails inundate inboxes and can pose security threats, waste time, and decrease productivity. The objective of this project is to create a reliable email classifier that distinguishes between legitimate "ham" emails and harmful "spam" emails. The model aims to achieve an accuracy score of 96% or higher on both training and testing data.

## Data
We use a publicly available email dataset that includes labeled "ham" and "spam" emails. This dataset can be found at [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset). It consists of email content and corresponding labels, which are used for training and testing our model.

## Prerequisites
To run this project, you'll need the following libraries installed:

- Scikit-Learn
- NumPy
- Pandas

You can install these libraries using pip:

```bash
pip install scikit-learn numpy pandas
```

## Installation
To get started with the **Ham-Spam Email Classifier**, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/ham-spam-email-classifier.git
```

## Usage
1. Open the Jupyter Notebook or Python script in your favorite development environment (e.g., Spyder).

2. Import the necessary libraries:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

3. Load the dataset:

```python
data = pd.read_csv("mail_data.csv")
```

4. Preprocess the data, split it into training and testing sets, and train the logistic regression model.

5. Use the model to make predictions and calculate accuracy.

6. Deploy the model using Streamlit and create an interactive interface for users to input email data for classification.

## Functions Explanation
### train_test_split
`train_test_split` is a function from Scikit-Learn that splits the dataset into training and testing sets. This is essential to evaluate the model's performance.

### accuracy_score
`accuracy_score` is a function from Scikit-Learn that measures the accuracy of the model's predictions. It calculates the percentage of correct predictions.

### Prediction
In machine learning, prediction refers to using a trained model to make forecasts or categorize new data points based on the patterns it has learned from the training data.

## Logistic Regression for Binary Classification
Logistic regression is a statistical method used for binary classification problems, where the goal is to predict one of two possible outcomes (in this case, "ham" or "spam"). It models the probability that a given input belongs to one of the two classes using a logistic function.

## Model Performance
Our model's performance can be assessed using various metrics, including accuracy, precision, recall, and F1-score. In this project, we focus on achieving a high accuracy score of 96% or above.

## Deployment
We use Anaconda to manage our Python environment, Streamlit to deploy the model, and Spyder as our integrated development environment. Streamlit allows us to create a user-friendly interface for users to input email data and get predictions instantly.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
