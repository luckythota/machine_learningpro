# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:00:25 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"E:\cpp programs\New folder\New folder\MetroPT3(AirCompressor).csv")
data.columns
data.shape
data.head

data.isna().sum()
data.drop('timestamp',axis=1,inplace=True)
data

x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

import pickle
sample=r"E:\cpp programs\New folder\New folder\MetroPT3(AirCompressor)final_model.sav"
pickle.dump(model,open(sample,'wb'))


import pickle
model_sample=pickle.load(open(r"E:\cpp programs\New folder\New folder\MetroPT3(AirCompressor)final_model.sav","rb"))
model_sample.predict([[0,-0.012,9.358,9.34,-0.024,9.358,53.6,0.04,1,0,1,1,0,1,1]])


#same as joblib   import joblib

