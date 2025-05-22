import numpy as np 
from numpy import linalg as LA

m, n = 3, 4
A = np.random.rand(m, n)
U, S, V = LA.svd(A) # A = U*S*V     (no V transpose here)
# checking if U, V are orthogonal (Trực giao) and S is a diagonal matrix with nonnegative decreasing elements
print('Frobenius norm of (UU^T - I) =', LA.norm(U.dot(U.T) - np.eye(m))) # Nếu U là ma trận orthogonal thì U.U^T = I  
print('S = ', S)
print('Frobenius norm of (VV^T - I) =', LA.norm(V.dot(V.T) - np.eye(n)))