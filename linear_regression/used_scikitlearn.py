from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Số lượng điểm dữ liệu
num_points = 100

# Tạo dữ liệu X ngẫu nhiên
X = np.random.uniform(0, 10, num_points)  # Giá trị X từ 0 đến 10

# Tạo noise ngẫu nhiên (nhiễu)
noise = np.random.normal(0, 1, num_points)  # Noise với phân phối chuẩn (mean=0, std=1)

# Tạo dữ liệu Y theo phương trình Y = 3X + 4 + noise
Y = 3 * X + 4 + noise

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
regress = linear_model.LinearRegression()
regress.fit(X_train.reshape(-1, 1), Y_train) # -1 là số hàng, mình muốn nói cho numpy biết là mình đang ko biết có bao nhiêu hàng
Y_pred = regress.predict(X_test.reshape(-1, 1))
# Tính MSE
mse = mean_squared_error(Y_test, Y_pred)
print(f"MSE: {mse}")
plt.scatter(X_test, Y_test, label='Dữ liệu gốc')
plt.scatter(X_test, Y_pred, label='Dự đoán', color='red')

# Vẽ đường hồi quy (dự đoán trên toàn bộ dải giá trị X)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Tạo dải X từ min đến max
Y_range = regress.predict(X_range)  # Dự đoán Y cho dải X
plt.plot(X_range, Y_range, color='black', label='Đường hồi quy')  # Vẽ đường hồi quy

plt.legend()
plt.show()
