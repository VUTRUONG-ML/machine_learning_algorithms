from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
n_samples = 1000
X1 = np.random.uniform(1, 1500, n_samples)
X2 = np.random.uniform(1, 1500, n_samples)

# Tạo noise ngẫu nhiên (nhiễu)
noise = np.random.normal(0, 0.1, n_samples)  # Noise với phân phối chuẩn (mean=0, std=1)

X = np.vstack([X1, X2]).T

y = 2*X1 + 0.5 * X2 + noise

# Tach du lieu 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tao mo hinh linear regression

li = LinearRegression()
li.fit(X_train, y_train)

y_pred_li = li.predict(X_test)

mse_li = mean_squared_error(y_test, y_pred_li)
print(f"Coefficient linear regression: {li.coef_}")
print(f"Intercepts: {li.intercept_}")
print(f"Mean Squared Error (MSE): {mse_li}")

# Lasso regression
las = Lasso(alpha=1.5)
las.fit(X_train, y_train)

y_pred_las = las.predict(X_test)
mse_las = mean_squared_error(y_test, y_pred_las)
print(f"Coefficient lasso regression: {las.coef_}")
print(f"Intercepts: {las.intercept_}")
print(f"Mean Squared Error (MSE): {mse_las}")

# Bieu do
plt.scatter(y_test, y_pred_li, color="blue", edgecolors="black")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression')
plt.show()