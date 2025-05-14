from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 
california_data = fetch_california_housing()

X = pd.DataFrame(california_data.data, columns= california_data.feature_names)
y = california_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressMod = LinearRegression()
regressMod.fit(X_train, y_train)
y_pred = regressMod.predict(X_test)

# In ra hệ số (coefficients) và intercept
print(f"Coefficients: {regressMod.coef_}")
print(f"Intercept: {regressMod.intercept_}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

#Ve bieu do
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color="blue", edgecolors='black', alpha=0.5) # Bieu do cham cham, tang xuat
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted Values')
plt.show()