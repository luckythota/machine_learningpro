# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:48:36 2023

@author: NAGA LAKSHMI
"""


import pandas as pd
import numpy as np

#reading the dataset
data=pd.read_csv(r'E:/cpp programs/New folder/New folder/Tetuan City power consumption.csv')
data
#data understanding
data.columns
data.shape
data.isna().sum()


#data preprocessing
data['day']=pd.to_datetime(data['DateTime']).dt.day
data['month']=pd.to_datetime(data['DateTime']).dt.month
data['years']=pd.to_datetime(data['DateTime']).dt.year
data['hours']=pd.to_datetime(data['DateTime']).dt.hour
data['minutes']=pd.to_datetime(data['DateTime']).dt.minute

data.drop('DateTime',axis=1,inplace=True)


#splitting the data to independent features and target features
x=data.iloc[:,:5 and 8:].values
y=data.iloc[: ,5:8].values

#model building
from sklearn.model_selection import train_test_split #testsize also not fixed
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)#random state is not fixed


#learing the algorithm
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)



#applying regression dataset to classification algorithm that is KNN
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)


#applying regression to dataset classification algorithm that is Decision tree

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)


#applying regression to dataset classification algorithm that is random forest
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(xtrain,ytrain)



#predicting the xtest values
ypred=model.predict(xtest)


#performance metrics : error
from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(ytest, ypred))








