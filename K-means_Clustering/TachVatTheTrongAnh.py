import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('E:/PythonSource/Data/anhthu.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

for K in [2, 5, 10, 15, 20]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k] # Thay the tat ca cac diem du lieu thuoc cum do bang chinh centroid cua no luon 
    # Thay doi size va hien thi anh ra ben ngoai
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation = 'nearest')
    plt.axis('off')
    plt.show()