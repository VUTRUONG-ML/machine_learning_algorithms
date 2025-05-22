import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cali_data = datasets.fetch_california_housing()

X = pd.DataFrame(data=cali_data.data, columns=cali_data.feature_names)
y = cali_data.target

#Tach du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Training size: ', X_train.shape[0], ', Test size: ', X_test.shape[0])

# Chuan hoa truoc khi build mo hinh
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Shape of X_train: {X_train.shape}")
print(f"Feature names: {cali_data.feature_names}")

k = 10
model = neighbors.KNeighborsRegressor(n_neighbors=k, p=2)

model.fit(X_train, y_train)
#Du doan
y_pred = model.predict(X_test)

#Danh gia
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

errors = []
k_values = range(1, 21)

for k in k_values:
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    errors.append(rmse)

# Vẽ biểu đồ RMSE theo k
plt.plot(k_values, errors, marker='o')
plt.title("RMSE vs. k")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()
