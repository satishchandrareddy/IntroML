# Chapter1.4_LinearAlgebra.py

import numpy as np

# transpose example 1
W = np.array([[1,2,3,4]])
print("W: {}".format(W))
print("W transpose: {}".format(W.T))

# transpose example 2
X = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("X: {}".format(X))
print("X transpose: {}".format(X.T))


# matrix multiplication example 1
W = np.array([[1,2,3]])
X = np.array([[4,7],[5,8],[6,9]])
Z = np.dot(W,X)
print("Matrix Multiplication Example 1")
print("W: {}".format(W))
print("X: {}".format(X))
print("Z: {}".format(Z))

# matrix multiplication example 2
W = np.array([[1,2,3],[2,3,4]])
X = np.array([[4,7],[5,8],[6,9]])
Z = np.dot(W,X)
print("Matrix Multiplication Example 2")
print("W: {}".format(W))
print("X: {}".format(X))
print("Z: {}".format(Z))

# broadcastng example 1
W = np.array([[1,2,3]])
X = np.array([[4,7],[5,8],[6,9]])
b = 7
Z = np.dot(W,X) + b
print("Broadcastng Example 1")
print("W: {}".format(W))
print("X: {}".format(X))
print("b: {}".format(b))
print("Z: {}".format(Z))

# broadcasting example 2
W = np.array([[1,2,3],[2,3,4]])
X = np.array([[4,7],[5,8],[6,9]])
b = np.array([[11],[12]])
Z = np.dot(W,X) + b
print("Broadcasting Example 2")
print("W: {}".format(W))
print("X: {}".format(X))
print("b: {}".format(b))
print("Z: {}".format(Z))