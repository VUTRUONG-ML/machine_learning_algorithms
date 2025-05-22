import numpy as np 
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Tải bộ dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial') 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy: ", 100*accuracy_score(y_pred, y_test))