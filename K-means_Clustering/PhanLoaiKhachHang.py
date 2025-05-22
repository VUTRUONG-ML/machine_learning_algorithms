import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
num_customer = 150
data = {
    "Age": [np.random.randint(18,70) for _ in range(num_customer)],
    "Annual Income": [np.random.randint(20000, 150000) for _ in range(num_customer)],
    "Spending score": [np.random.randint(1, 150) for _ in range(num_customer)],
}

customer_df = pd.DataFrame(data)

print(customer_df)
print("ok")

X = customer_df

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
customer_df["Cluster"] = kmeans.labels_
conditions = [
    customer_df["Cluster"] == 0,
    customer_df["Cluster"] == 1,
    customer_df["Cluster"] == 2
]
choices = ["low", "medium", "high"]

customer_df["Group"] = np.select(conditions, choices,default="unknown")
print(kmeans.cluster_centers_)
print(customer_df)