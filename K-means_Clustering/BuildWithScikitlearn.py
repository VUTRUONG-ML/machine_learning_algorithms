import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # Tinh khoang cach giua cac cap diem trong 2 tap hop
from sklearn.cluster import KMeans
np.random.seed(42)
means = [[2,2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N) # các điểm dữ liệu có 2 cột, giá trị xung quanh 2,2 có phương sai là 1, 500 hàng 
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0) # ghép các điểm dữ liệu lại với nhau theo cột 
K = 3 # 3 cluster
original_label = np.asanyarray([0]*N + [1]*N + [2]*N).T # ma trận các nhãn 0,1,2

# Khoi tao cac centroids
def kmeans_init_centroids(X, k):
    # chon ngau ngun k hang lam centroid trong X
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Tim label moi cho cac diem du lieu khi co dinh centroid 
def kmeans_assign_labels(X, centroids):
    # Tinh khoang cach giua tung cap du lieu - centroids
    D = cdist(X, centroids) # D[i][j] la khoang cach tu Xi đến centroid[j]
    # tra ve chi so cua tam gan nhat
    return np.argmin(D, axis=1) # tra ve mang 1D chua cac chi so cua centroid gan nhat voi diem du lieu tuong ung 

# Kiem tra dieu kien dung cua kmeans clustering
def has_converged(centroids, new_centroids):
    # Tra ve true neu hai centroi giong nhau 
    return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]))

# Cap nhat cac centroids khi biet label cua moi diem du lieu
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        # Lay all cac diem duoc gan vao cluster thu k
        Xk = X[labels == k, : ]
        centroids[k, : ] = np.mean(Xk, axis=0) # Tinh trung binh 
    return centroids 

def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids): break
        centroids.append(new_centroids)
        it += 1
    return (centroids, labels, it)

(centroids, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:\n', centroids[-1])

print()
print("With scikitlearn")
model = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(model.clustere_centers_)
pred_label = model.predict(X)
print('Predicted labels by scikit-learn:', pred_label)