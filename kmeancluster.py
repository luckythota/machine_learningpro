# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:24:47 2023

@author: NAGA LAKSHMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel(r"E:\cpp programs\New folder\New folder\kmeans1.xlsx")
data
data.shape
data.columns

data1=data.drop(['ID Tag','Model'],axis=1)
data1=data1.drop(['Department'],axis=1)

data.head()

from sklearn.cluster import KMeans
model=KMeans(n_clusters=4,init='k-means++',n_init=10)
model.fit(data1)


x=model.fit_predict(data1)
print(x)

data['cluster']=x
data1=data.sort_values(['cluster'])
data1

data1.to_csv(r"E:\cpp programs\New folder\New folder\kmeanspredicted.csv")



