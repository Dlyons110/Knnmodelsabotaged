# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:35:48 2023
@author: David
"""
title: "Assigment - kNN DIY"
 # - name author here - David
 # - name reviewer here -
 "`r format(Sys.time(), '%d %B, %Y')`"
output:
   html_notebook:
    toc: true
    toc_depth: 
---
pip install pandas
pip install tidyverse
pip install googlesheets4
pip install class
pip install caret
import pandas as pd
from pandas import CategoricalDtype
import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Choose a suitable dataset from [this](https://github.com/HAN-M3DM-Data-Mining/assignments/tree/master/datasets) folder and train  your own kNN model. Follow all the steps from the CRISP-DM model.

#Business Understanding#
Diabetes is a medical disorder that has an impact on how well your body uses food as fuel.
The majority of the food you consume is converted by your body into sugar (glucose), which is then released into your bloodstream. 
Your pancreas releases insulin when your blood sugar levels rise.
With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. 
There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help.
Source: https://www.cdc.gov/diabetes/basics/diabetes.html#:~:text=Diabetes%20is%20a%20chronic%20(long,your%20pancreas%20to%20release%20insulin.

#Data Understanding#
url = "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/KNN-diabetes.csv"
rawDF = pd.read_csv('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/KNN-diabetes.csv')
rawDF.info()

testDF = rawDF[(rawDF.Insulin !=0) & (rawDF.BMI !=0)&(rawDF.SkinThickness !=0)&(rawDF.BloodPressure !=0)]
#Removed the zeroes in the dataset to make sure that the zeroes are removed in places where they shouldn't or can't be

cleanDF = testDF.drop(['BloodPressure'], axis=1) #Changed rawDF to testDF

#BloodPressure not influence if you have diabetes or not
cleanDF.head()
rawDF['Outcome'] = rawDF['Outcome'].replace([0]) 
rawDF['Outcome'] = rawDF['Outcome'].replace([1]) 
#the D represents a diagnosis of diabetes
#the N represents a diagnosis of no diabetes

#Data Preparation#
cntDiag = testDF['Outcome'].value_counts() 
propDiag = testDF['Outcome'].value_counts(normalize=True)
cntDiag #the outcome is in this case the same as the diagnosis, so to see if someone has diabetes or not
propDiag 

cleanDF.info()

catType = CategoricalDtype(categories=[0,1], ordered=False)
cleanDF['Outcome'] = cleanDF['Outcome'].astype(catType)
cleanDF['Outcome']
cleanDF['Outcome'].unique()
cleanDF[['Pregnancies', 'Insulin', 'BMI']].describe()

def normalize(x):
  return((x - min(x)) / (max(x) - min(x))) # distance of item value - minimum vector value divided by the range of all vector values

testSet1 = np.arange(1,6)
testSet2 = np.arange(1,6) * [[10]]

print(f'testSet1: {testSet1}\n')
print(f'testSet2: {testSet2}\n')
print(f'Normalized testSet1: {normalize(testSet1)}\n')
print(f'Normalized testSet2: {normalize(testSet2)}\n') 

excluded = ['Outcome'] # list of columns to exclude
X = cleanDF.loc[:, ~cleanDF.columns.isin(excluded)]
X = X.apply(normalize, axis=0)

X[['Pregnancies', 'Insulin', 'BMI']].describe() #Changed the columns into relative ones

y = cleanDF['Outcome'] #fixed Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

#Modeling#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) #makes predictions based on the test set

y_pred = knn.predict(X_test)
keepdims = True
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
cm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

#Calculate metrics#
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Print metrics#
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")

#Data Evaluation#
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.show() 