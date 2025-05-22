import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load
data = load_iris()
x = data.data 
y = data.target
target_names = data.target_names  # ['setosa', 'versicolor', 'virginica']

# scaler 
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

# PCA , giam số chiều xuống còn k 
k = 2 # tại sao chọn k bằng 2 quay sang đoạn code tính explained_variance_ratio_
pca = PCA(n_components=k)
X_pca = pca.fit_transform(x_scaler)

# 4. Vẽ biểu đồ scatter với chú thích màu
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                label=target_name, color=colors[i], edgecolor='k')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA trên tập dữ liệu Iris")
plt.legend()  # Thêm chú thích
plt.show()