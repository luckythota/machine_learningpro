# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:52:11 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#header names mentioned in the list a
a=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']
#header names applied to loaded dataset 
data=pd.read_csv(r"E:\cpp programs\New folder\New folder\income.csv",names=a)
#to print data
data
#to print the dimensions of the dataset
data.shape
#to print the columns of the dataset
data.columns
#to check wheather empty spaces are there or not each and every column in dataset
data.isna().sum()

data['capital_gain'].value_counts()
data['capital_loss'].value_counts()
data['relationship'].value_counts()

#to drop the columns which are not required for prediction in the dtaset
data.drop(['fnlwgt','relationship','capital_gain','capital_loss'],axis=1,inplace=True)
#check whether the above columns are deleted or not
data.columns


#to check the count for each age distribution whether unnecessary values are there or not
b=data['age'].value_counts()
#to check whether nan or unnecessary values are there or not
data['workclass'].value_counts()
#to count the most repeated value in column workclass
m=data['workclass'].mode()
#replacing ? with mode value
data['workclass'].replace(to_replace=' ?',value=' Private',inplace=True)
#for all columns to check whether unnecessary values or nan is present or not
data['education'].value_counts()
data['education_num'].value_counts()
data['marital_status'].value_counts()
data['occupation'].value_counts()
#replacing ? with most repeated occupation in the dataset
data['occupation'].replace(to_replace=' ?',value= data['occupation'].mode()[0],inplace=True)
data['race'].value_counts()
data['sex'].value_counts()
hr=data['hours_per_week'].value_counts()
data['native_country'].value_counts()
#replacing ? with most repeated value in native country in the dataset
data['native_country'].replace(to_replace=' ?',value= data['native_country'].mode()[0],inplace=True)
data['income'].value_counts()

#to convert all columns in string form to number form
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    data[i]=le.fit_transform(data[i])


#divide the data into  independent features i.e x and target feature i.e. y
x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values


#split the dtaset into training and testing and testing size is 20%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)

#importing the KNeighbors algorithm from sklearn  
from sklearn.neighbors import KNeighborsClassifier
# set the k=4  and place the model 
model=KNeighborsClassifier(n_neighbors=4)
#fit the model using xtrain and ytrain at this time algorithm starts learning
model.fit(xtrain,ytrain)
#then algorithm predicts basing on testing data i.e xtest
ypred=model.predict(xtest)


#import the accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score
#calculating the accuracy using ytest and ypred
accuracy_score(ytest,ypred)*100

#for predicting the values
model.predict([[22,5,6,12,2,3,4,1,44,22]])
model.predict([[65,7,5,11,2,5,5,2,45,23]])


model.predict([[34,4,5,6,9,10,23,45,6,9]])


