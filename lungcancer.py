# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:00:40 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"E:\cpp programs\New folder\New folder\survey lung cancer.csv")
data
data.columns
data.shape
data.isna().sum()

#data["GENDER"]=data["GENDER"].map({"M" : 0, "F":1}) #to convert string to number
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()                              # to convert string to number we have multiple categories in that string
data['GENDER']=le.fit_transform(data['GENDER'])

x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values


from sklearn.model_selection import train_test_split #testsize also not fixed
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1) #random state is not fixed


 
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100







