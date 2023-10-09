# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:27:29 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\NAGA LAKSHMI\Downloads\co\iris.csv")


data.shape
data.columns
data
data.head()

data['vl']=np.arange(0,150) #adding vl column

data.isna().sum()

data['pl'].fillna(data['pl'].mode()[0],inplace=True)
#data.dropna(inplace=True)



a=data.corr() #correlation matrix
plt.matshow(a) #plotting a correlation
plt.colorbar()
plt.show()



#we have to divide independent features and target features 
x=np.array(data.iloc[ : ,[0,1,2,3,5]])#to convert in form of array to train a algorithm and it's for independent features
y=data.iloc[ : ,-2].values#to convert in form of array to train a algorithm either we use np.array or .values  and it's for target features
x.shape
y.shape


from sklearn.model_selection import train_test_split #testsize also not fixed
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1) #random state is not fixed


#from sklearn.model_selection import train_test_split
xtrain.shape #shape of xtrain
ytrain.shape#shape of ytrain 1.3 1.5
xtest.shape#shape of xtest
ytest.shape#shape of ytest


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)
