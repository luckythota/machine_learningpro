# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:05:17 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
a=[i for i in 'abcdefghijklmnop']
data=pd.read_csv(r"E:\cpp programs\New folder\New folder\credit_loan.csv",names=a)
data.columns
data.shape

data['a'].replace(to_replace='?',value=data['a'].mode()[0],inplace=True)
data['b'].replace(to_replace='?',value=10,inplace=True)
data['d'].replace(to_replace='?',value=data['d'].mode()[0],inplace=True)
data['e'].replace(to_replace='?',value=data['e'].mode()[0],inplace=True)
data['f'].replace(to_replace='?',value=data['f'].mode()[0],inplace=True)
data['g'].replace(to_replace='?',value=data['g'].mode()[0],inplace=True)
data['n'].replace(to_replace='?',value=data['n'].mode()[0],inplace=True)


data['a'].value_counts()
data['b'].value_counts()
data['c'].value_counts()
data['d'].value_counts()
data['e'].value_counts()
data['f'].value_counts()
data['g'].value_counts()
data['h'].value_counts()
data['i'].value_counts()
data['j'].value_counts()
data['k'].value_counts()
data['l'].value_counts()
data['m'].value_counts()
data['n'].value_counts()
data['o'].value_counts()
data['p'].value_counts()


data.isna().sum()

s='adefgijlm'
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in s:
    data[i]=le.fit_transform(data[i])


x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100

model.predict([[2,124,45,1,1,13,7,73,0,0,0,2,1,4,2]])

