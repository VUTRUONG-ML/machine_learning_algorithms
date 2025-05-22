import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
iris = load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Tạo DataFrame chỉ chứa hai loại hoa Setosa (0) và Versicolor (1)
data_two_class = data[data['species'].isin([0, 1])]

X = data_two_class.iloc[:, :-1].values
y = data_two_class['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(C= 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy :{100*accuracy_score(y_test, y_pred)}")
