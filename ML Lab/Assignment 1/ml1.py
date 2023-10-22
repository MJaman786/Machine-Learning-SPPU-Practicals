# -*- coding: utf-8 -*-
"""ML1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1txzQycchSYlZZMZCNAqZr3AFJinh5VsT
"""

# to perform structural data analysis
import pandas as pd

df = pd.read_csv("/content/Heart.csv")

df.head()

df.shape

# finding missing value
df.isnull()

df.isnull().sum()

# datatypes of each column
df.dtypes

# find out zeros
df == 0

df[df==0].count()

df.columns

# finding mean
df['Age'].mean()

# This is called label based slicing
newdf = df[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']]

print(newdf)

# performing cross validation
# training and testing

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=0, train_size=0.75)

train.shape

test.shape

import numpy as np

actual = np.concatenate((np.ones(45),np.zeros(450),np.ones(5)))

print(actual)

predicted = np.concatenate((np.ones(100),np.zeros(400)))

print(predicted)

# displaying confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(actual, predicted)

# importing lib for calculation
  # Accuracy
  # Precision
  # Recall
  # F-1 score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(actual, predicted))

accuracy_score(actual,predicted)