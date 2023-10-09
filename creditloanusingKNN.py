# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:21:17 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=[i for i in 'abcdefghijklmnop']
data=pd.read_csv(r"E:\cpp programs\New folder\New folder\credit_loan.csv",names=a)
print(data.columns)
##a=np.arange(1,151)
##data['new']=a

print(data.shape)
#print(data)
print(data.head())

print(data.isna().sum())


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


s='adefgijlm'
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in s:
    data[i]=le.fit_transform(data[i])

x=np.array(data.iloc[ : , :-1])##[0,1,2,3,5]])
y=data.iloc[ : ,-1].values
print(x.shape,y.shape)

#scaling is especially used for knn because in case of a lot difference between values in a column i.e outliers occurence
#scaling=x-min(x)/max(x)-min(x)
#normalization-->minmaxscaler
#standardization-->standardization
#scaling-->standardscaler
#y=x-mean/sd



#normalization
from sklearn.preprocessing import MinMaxScaler
scaled=MinMaxScaler()
scaledx=scaled.fit_transform(x)



#standardization
from sklearn.preprocessing import StandardScaler
scaled=StandardScaler()
scaledx=scaled.fit_transform(x)




from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(scaledx,y,test_size=0.3,random_state=1)

print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)



from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

model.predict([x[0]])


