# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:25:09 2023

@author: NAGA LAKSHMI
"""

#linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=[i for i in 'abcdefghijklmn']
data=pd.read_csv(r"E:\cpp programs\New folder\New folder\boston.csv",names=a)
print(data)

data.columns

data.isna().sum()
data.dropna(inplace=True)

x=np.array(data.iloc[ : , :-1])
y=data.iloc[ : ,-1].values


#from sklearn.preprocessing import OneHotEncoder
#o_hot=OneHotEncoder()
#df=pd.DataFrame(o_hot.fit_transform(data[['workclass']]).toarray())
#data=data.join(df)
#print(data)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)


from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(ytest, ypred))












#from sklearn.linear_model import LogisticRegression
#model=LogisticRegression()



