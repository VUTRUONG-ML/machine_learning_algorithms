import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

num_item = 200

np.random.seed(42)

distance = np.random.randint(1, 100, num_item) # kilometer
weight = np.random.uniform(1, 20, num_item) # kilogam
priority = np.random.choice(["high", "low"], num_item) # 
delivery_time = distance/10 + np.random.normal(0,2, num_item)

df = pd.DataFrame({
    'Distance': distance,
    'Weight': weight,
    'Priority': priority,
    'Delivery Time': delivery_time
})
df['Priority'] = df['Priority'].map({'high': 1, 'low': 0})
X = df[['Distance', 'Weight', 'Priority']]

y = df['Delivery Time']

# Tach du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuan hoa
scaler = StandardScaler()
X_train[['Distance', 'Weight']] = scaler.fit_transform(X_train[['Distance', 'Weight']])
X_test[['Distance', 'Weight']] = scaler.transform(X_test[['Distance', 'Weight']])

# Du doan
k = 3
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE : {mse}")

k_values = range(1, 21)

for i in k_values:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(rmse)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'red')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted Values')
plt.show()

