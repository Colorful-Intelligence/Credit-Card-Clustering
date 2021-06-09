#%% Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-dark-palette")

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

#%% Read the dataset

data = pd.read_csv("credit_card_dataset.csv")


#%% Exploratory Data Analysis (EDA)

data.columns

"""
['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
"""

data.shape # (8950, 18)



data.info()

data.drop(["CUST_ID"],axis = 1,inplace = True)

#%% Missing Values
data.columns[data.isnull().any()]
data.isnull().sum()

#%% Filling Missing Values

data["MINIMUM_PAYMENTS"] = data["MINIMUM_PAYMENTS"].fillna(data["MINIMUM_PAYMENTS"].mean())
data["CREDIT_LIMIT"] = data["CREDIT_LIMIT"].fillna(data["CREDIT_LIMIT"].mean())

#%% Standardize Data
from sklearn.preprocessing import StandardScaler , normalize

scaler = StandardScaler()
scaler_df = scaler.fit_transform(data)

#%% Normalizing Data 
normalized_df = normalize(scaler_df)

#%% PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2']


#%% finding the optimum number of clusters for k-means algorithm 

from sklearn.cluster import KMeans

wcss=[] #with in cluster sum of square error
for i in range(1,11):
    kmeans =KMeans(n_clusters = i, init = 'k-means++', max_iter=200, n_init=10,random_state=0)
    kmeans.fit(X_principal)
    wcss.append(kmeans.inertia_)



#ploting the results onto a line graph
#Elbow method
plt.plot(range(1,11),wcss)
plt.title('THE ELBOW METHOD')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_principal)

plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = KMeans(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter,alpha = 0.5) 

plt.show() 

#plotting the centroid of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 100, c= 'red', label = 'centroids')
plt.title('K-means For Credit Card')

plt.legend()

