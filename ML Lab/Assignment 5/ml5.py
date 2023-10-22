
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Mall_Customers.csv')

df.head()

df.shape

df.columns

df.dtypes

df.isnull().sum()

x = df.iloc[:,3:]

x

plt.title('Uncluster Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])

"""# 1. Method Elbow Method KMeans Algorithm"""

from sklearn.cluster import KMeans, AgglomerativeClustering

# passing how many cluster you want to create
km = KMeans(n_clusters=3)

x.shape

km.fit_predict(x)

# Sum Square Error(SSE)
km.inertia_

sse = []
for k in range(1,16):
  km = KMeans(n_clusters=k)
  km.fit_predict(x)
  sse.append(km.inertia_)

sse

plt.title('Elbow - Metod')
plt.xlabel('value of "k"')
plt.ylabel('SSE')
plt.grid()
plt.xticks(range(2,16))
plt.plot(range(1,16), sse, marker='.', color='blue')

"""# 2. Method Silhoute Score"""

from sklearn.metrics import silhouette_score

silh = []
for k in range(2,16):
  km = KMeans(n_clusters=k)
  labels = km.fit_predict(x)
  score = silhouette_score(x, labels)
  silh.append(score)

silh

plt.title('Silhoutte - Metod')
plt.xlabel('value of "k"')
plt.ylabel('Silhoutte Score')
plt.grid()
plt.xticks(range(2,16))
plt.bar(range(2,16), silh, color='red')

km = KMeans(n_clusters=5)

labels = km.fit_predict(x)

labels

km.cluster_centers_

cent = km.cluster_centers_

plt.figure(figsize=(16,9))
plt.subplot(1,2,1)
plt.title('Uncluster Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])

plt.subplot(1,2,2)
plt.title('Clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=labels)
plt.scatter(cent[:,0], cent[:,1], s=100, color='black')

# this is cluster 4
cluster_4 = df[labels==4]
cluster_4

km.predict([[56,61]])