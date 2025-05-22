import tensorflow as tf
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Tải bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train0 = x_train[np.where(y_train == 0)[0]]
X_train1 = x_train[np.where(y_train == 1)[0]]
y_train0 = np.zeros(X_train0.shape[0])
y_train1 = np.ones(X_train1.shape[0])

X_test0 = x_test[np.where(y_test == 0)[0]]
X_test1 = x_test[np.where(y_test == 1)[0]]
y_test0 = np.zeros(X_test0.shape[0])
y_test1 = np.ones(X_test1.shape[0])

X_train = np.concatenate((X_train0, X_train1), axis = 0)
X_test = np.concatenate((X_test0, X_test1), axis = 0)
Y_train = np.concatenate((y_train0, y_train1))
Y_test = np.concatenate((y_test0, y_test1))

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# Kiểm tra thông tin dữ liệu
print("Dữ liệu huấn luyện:", X_train.shape, Y_train.shape)
print("Dữ liệu kiểm tra:", X_test.shape, Y_test.shape)

model = LogisticRegression(C= 1e5) # inverse of lam
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


print(f"Accuracy {100*accuracy_score(Y_test, Y_pred)}")

import matplotlib.pyplot as plt

# Tìm các chỉ số của mẫu bị phân loại sai
mis = np.where(Y_pred != Y_test)[0]

# Hiển thị các hình ảnh bị phân loại sai
print(f"Số lượng hình bị phân loại sai: {len(mis)}")

# Hiển thị tối đa 10 hình bị phân loại sai
num_to_display = min(len(mis), 10)

for i in range(num_to_display):
    index = mis[i]
    image = X_test[index].reshape(28, 28)  # Chuyển đổi lại thành ảnh 28x28
    true_label = int(Y_test[index])
    predicted_label = int(Y_pred[index])
    
    plt.imshow(image, cmap="gray")
    plt.title(f"Dự đoán: {predicted_label}, Thực tế: {true_label}")
    plt.axis("off")
    plt.show()
