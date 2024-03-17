#!/usr/bin/env python
# coding: utf-8

# In[24]:


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the CSV file
data = pd.read_csv('E:\SAHAANA NEW\Datasets\Mall_Customers.csv')
print(d.head())

# Separate numeric and categorical features
numeric_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(exclude=[np.number])

# One-hot encode categorical features
encoder = OneHotEncoder()
categorical_features_encoded = encoder.fit_transform(categorical_features).toarray()

# Combine encoded categorical features with numeric features
encoded_data = np.concatenate((categorical_features_encoded, numeric_features), axis=1)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(encoded_data)

# Define the number of clusters (adjust as needed)
num_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_data)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Print cluster labels and centroids
print("Cluster Labels:")
print(cluster_labels)
print("\nCentroids:")
print(centroids)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(d[['CustomerID']],d['Annual Income (k$)'])
plt.xlabel('CUSTOMER ID')
plt.ylabel('SPENDING SCORE')
plt.title('Before Clustering')

plt.subplot(1, 2, 2)
plt.scatter(d[['Annual Income (k$)']],d['Spending Score (1-100)'], c=cluster_labels)
plt.legend(cluster_labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('ANNUAL INCOME')
plt.ylabel('SPENDING SCORES')
plt.title('After Clustering')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




