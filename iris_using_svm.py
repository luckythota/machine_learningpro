# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:19:26 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np

data=pd.read_csv(r'C:\Users\NAGA LAKSHMI\Downloads\co\iris.csv')
data

data.shape
data.columns

data.isna().sum()


x=data.iloc[: ,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.svm import SVC   #support vector classifier
model=SVC(kernel='Linear')#some other functions like poly, rbf if we apply kernel ,kernel is used for non linear separable data for linear 
#we directly use model=SVC()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100


