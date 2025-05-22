import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data 
mean_x = np.mean(X, axis = 0)
X_centered = X - mean_x

# Tinh ma tran hiep phuong sai 
# tại sao phải tính ma trận hiệp phương sai -> để tìm sự sai khác nhau ra sao giữa các đặc trưng 
# cái phương sai cũng là độ phân tán của dữ liệu 
cov_matrix = np.cov(X_centered, rowvar=False)
print("Ma trận hiệp phương sai:\n", cov_matrix)

# eigenvalues và eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Tính explained_variance_ratio_
# Eigenvector chỉ định hướng dữ liệu trải rộng theo chiều nào. ( vector riêng là một đường thẳng ko có chiều dài)
# Eigenvalue chính là thứ đo độ trải rộng theo hướng đó (cũng gần như là độ dài của phương sai theo hướng đó).

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

print("Explained Variance Ratio:\n", explained_variance_ratio)
print("Tổng phương sai giữ lại:", np.sum(explained_variance_ratio))

# --> rồi tóm lại là PCA sẽ chọn ít nhất là 2 phương sai mà nó dài nhất hay rộng nhất để giữ lại, còn nhưng phương sai khác bé quá thì sẽ bỏ đi
# Từ phương sai đó sẽ phản ánh dữ liệu xuống cái phương hướng đó và cho ra các giá trị khác 