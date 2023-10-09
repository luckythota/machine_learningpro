# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:14:54 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np

data=pd.read_csv(r'E:/cpp programs/New folder/New folder/datanew.csv')
data
data.shape
data.columns
data.isna().sum()
data.drop(['spore-print-color','veil-color','veil-type','stem-root'],axis=1,inplace=True)
data['cap-surface'].value_counts()
data['cap-surface'].fillna(data['cap-surface'].mode()[0],inplace=True)
data['gill-attachment'].value_counts()
data['gill-attachment'].fillna(data['gill-attachment'].mode()[0],inplace=True)
data['gill-spacing'].value_counts()
data['gill-spacing'].fillna(data['gill-spacing'].mode()[0],inplace=True)
data['stem-surface'].value_counts()
data['stem-surface'].fillna(data['stem-surface'].mode()[0],inplace=True)
data['ring-type'].value_counts()
data['ring-type'].fillna(data['ring-type'].mode()[0],inplace=True)


s=['class','cap-shape','cap-surface','cap-color','does-bruise-or-bleed','gill-attachment','gill-spacing','gill-color','stem-surface','stem-color','has-ring','ring-type','habitat','season']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in s:
    data[i]=le.fit_transform(data[i])



x=np.array(data.iloc[ : ,1 :])##[0,1,2,3,5]])
y=data.iloc[ : ,0].values
x.shape
y.shape


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)



#logistic regression:
    

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#knaivebayes classifier

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#decision tree

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)



