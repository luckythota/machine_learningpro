

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"E:\cpp programs\New folder\New folder\spotify-2023.csv")

data
data.columns
data.head()
data.isna().sum()
data['in_shazam_charts'].isna().count()
data['in_shazam_charts'].fillna(data['in_shazam_charts'].mode()[0],inplace=True)

data['in_shazam_charts'].value_counts()
data['in_shazam_charts'].isna().sum()
data['key'].value_counts()
data['key'].isna().sum()

data['key'].fillna(data['key'].mode()[0],inplace=True)


a=data.corr()
plt.matshow(a)
plt.colorbar()
plt.show()

b=data['released_year'].value_counts()
c=b.to_dict()
keys=list(c.keys())
values=list(c.values())
fig=plt.figure(figsize=(10,5))
k=plt.bar(keys, values,width=0.8,color=c)
plt.xlabel("year")
plt.ylabel("no. of movies")
plt.title("no.of movies in each year")
plt.show()


c=data['released_month'].value_counts()

d=data[data['released_month']==1].groupby('in_spotify_playlists').value_counts()


s=data['track_name'].value_counts()
import matplotlib.pyplot as plt
k=s.to_dict()
keys=list(k.keys())
values=list(k.values())
fig=plt.figure(figsize=(10,7))
plt.pie(values,labels=keys)
plt.show()

k=data['energy_%'].value_counts()
data['track_name'].value_counts().head(5)
data['in_apple_playlists'].groupby(data['track_name']).value_counts().head(5)






