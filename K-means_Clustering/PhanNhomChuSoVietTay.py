import numpy as np
from sklearn.datasets import fetch_openml
data_dir = r'E:\PythonSource\Data'
mnist = fetch_openml('mnist_784', data_home=data_dir)
print("Shape of mnist data: ", mnist.data.shape)