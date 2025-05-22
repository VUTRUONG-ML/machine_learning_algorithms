import numpy as np 
import imageio.v2 as imageio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42)

# filename structure
path = 'E:/PythonSource/unpadded/'   # path to the database
ids = range(1, 16) # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
'noglasses', 'normal', 'rightlight','sad',
'sleepy', 'surprised', 'wink' ]
prefix = 'subject'
surfix = '.pgm'

# data dimension
h, w, K = 116, 98, 100 # hight, weight, new dim
D = h * w
N = len(states)*15

#collect all data 
X = np.zeros((D, N))
cnt = 0 
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        img = imageio.imread(fn) 
        X[:, cnt] = img.reshape(D)
        cnt += 1

# Doing PCA, note that each row is a datapoint

pca = PCA(n_components= K)
pca.fit(X.T)

U = pca.components_.T

for i in range(U.shape[1]):
    plt.axis('off')
    f1 = plt.imshow(U[:, i].reshape(116, 98), interpolation='nearest')
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
#     f2 = plt.imshow(, interpolation='nearest' )
    plt.gray()
    fn = 'eigenface' + str(i).zfill(2) + '.png'
#     plt.savefig(fn, bbox_inches='tight', pad_inches=0)
    plt.show()