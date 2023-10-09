# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:25:31 2023

@author: NAGA LAKSHMI
"""

#classification example
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


from sklearn.neighbors import KNeighborsClassifier #importing the KNN algorithm
model=KNeighborsClassifier(n_neighbors=3) #n_neighbours is not fixed value
model.fit(xtrain,ytrain) #algorithm will learn from xtrain and ytrain
ypred=model.predict(xtest) #for predicting the y basing on xtest

from sklearn.metrics import accuracy_score #for accuracy importing accuracy_score
accuracy_score(ytest,ypred)*100 #calculate accuracy

model.predict([[5.6,3.2,2.1,0.8]])
model.predict([[4.8,3,1.4,0.1]])
model.predict([[6.1,2.9,4.7,1.4]])
model.predict([[6.4,2.7,5.3,1.9]])



#0-100
#1-100
#mode-100

