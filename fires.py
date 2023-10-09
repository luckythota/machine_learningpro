# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:34:16 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'E:\cpp programs\New folder\New folder\fires.csv')
data
data.columns
data.shape
data.isna().sum()
data.dropna(inplace=True)
data.drop(126,axis=0,inplace=True)
data.shape

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['day']=le.fit_transform(data['day'])
data['month']=le.fit_transform(data['month'])
data['year']=le.fit_transform(data['year'])
data['Temperature']=le.fit_transform(data['Temperature'])
data[' RH']=le.fit_transform(data[' RH'])
data[' Ws']=le.fit_transform(data[' Ws'])
data['Rain ']=le.fit_transform(data['Rain '])
data['FFMC']=le.fit_transform(data['FFMC'])
data['DMC']=le.fit_transform(data['DMC'])
data['DC']=le.fit_transform(data['DC'])
data['ISI']=le.fit_transform(data['ISI'])
data['BUI']=le.fit_transform(data['BUI'])
data['FWI']=le.fit_transform(data['FWI'])

x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=2)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100


