# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:36:06 2023

@author: NAGA LAKSHMI
"""

#logistic regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\NAGA LAKSHMI\Downloads\co\iris.csv")

data.isna().sum()

x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score #for accuracy importing accuracy_score
accuracy_score(ytest,ypred)*100 



#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(ytest,ypred)

