from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
# Số lượng điểm dữ liệu
n_samples = 10000

# Tạo dữ liệu X ngẫu nhiên
X1 = np.random.uniform(0, 100, n_samples)
X2 = np.random.uniform(0, 100, n_samples)
X3 = np.random.uniform(0, 100, n_samples)

# Tạo noise ngẫu nhiên (nhiễu)
noise = np.random.normal(0, 1, n_samples)  # Noise với phân phối chuẩn (mean=0, std=1)

# Tạo dữ liệu Y 
X = np.vstack([X1, X2, X3]).T
Y = 5 * X1 + 3 * X2 - 2 * X3 + noise

# Tach du lieu 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tao mo hinh Linear regression
reModel = linear_model.LinearRegression()

# Huan luyen
reModel.fit(X_train, Y_train)

#Du doan
Y_pred = reModel.predict(X_test)
# In kết quả
print("Linear Regression Coefficients:", reModel.coef_)

# Tao mo hinh Ride regression 
## chon gia tri lamda
lamda = 150.0
ridge_mod = Ridge(alpha=lamda)
## Huan luyen mo hinh
ridge_mod.fit(X_train, Y_train)
## Du doan ridge
Y_pred_ridge = ridge_mod.predict(X_test)
print("Ridge Regression Coefficients (alpha = {}):".format(lamda), ridge_mod.coef_)

# Vẽ đồ thị so sánh
plt.figure(figsize=(12, 6))

# Vẽ Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(Y_test, Y_pred)
plt.title("Linear Regression")
plt.xlabel("True Values")
plt.ylabel("Predictions")

# Vẽ Ridge Regression
plt.subplot(1, 2, 2)
plt.scatter(Y_test, Y_pred_ridge)
plt.title("Ridge Regression (alpha = {})".format(lamda))
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.tight_layout()
plt.show()
