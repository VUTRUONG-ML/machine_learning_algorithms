from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


np.random.seed(42)
iris_data = datasets.load_iris()
iris_X = iris_data.data
iris_Y = iris_data.target
print('Label:', np.unique(iris_Y))

#Tach du lieu training - test
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=130)
print('Training size: ', X_train.shape[0], ', Test size: ', X_test.shape[0])
k = 7
model = neighbors.KNeighborsClassifier(n_neighbors=k, p = 2) #p = 1: Euclidean distance, p = 2: Euclidean distance squared, n_neiighbors = k
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of ",k,"NN: %.2f %%" % (accuracy_score(y_test, y_pred) * 100))  #Accuracy : Tỉ lệ dự đoán đúng 

# k_values = range(1, 20)
# accuracies = []
# # Ve bieu do de tim k cho model KNN 
# for k in k_values:
#     model = neighbors.KNeighborsClassifier(n_neighbors=k, p=2)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracies.append(accuracy_score(y_test, y_pred))

# plt.plot(k_values, accuracies, marker='o')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs k')
# plt.show()

model = neighbors.KNeighborsClassifier(n_neighbors=7, p = 2, weights= 'distance') # weight = 'distance' : tính trọng số dựa trên khoảng cách, weight = 'uniform' : tính trọng số bằng nhau
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy of 7NN (1/distance weights): %.2f %%" % (100*accuracy_score(y_test,y_pred)))

# Tu tao ra trong so rieng
def myweight(distances):
    sigma2 = .4
    return np.exp(-distances**2/sigma2)

model = neighbors.KNeighborsClassifier(n_neighbors=7, p = 2, weights=myweight) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 7NN (customized weights): %.2f %%"%(100*accuracy_score(y_test, y_pred)))