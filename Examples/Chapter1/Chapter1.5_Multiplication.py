# Chapter1.5_Multiplication.py

import numpy as np

# matrix multiplication example 1
# W = [1 2 3]
# X = [4 7]
#     [5 8]
#     [6 9]
W = np.array([[1,2,3]])
X = np.array([[4,7],[5,8],[6,9]])
Z = np.dot(W,X)
print("Matrix Multiplication Example 1")
print("W: \n{}".format(W))
print("X: \n{}".format(X))
print("Z: \n{}".format(Z))

# matrix multiplication example 2
# W = [1 2 3]
#     [2 3 4]
# X = [4 7]
#     [5 8]
#     [6 9]
W = np.array([[1,2,3],[2,3,4]])
X = np.array([[4,7],[5,8],[6,9]])
Z = np.dot(W,X)
print("Matrix Multiplication Example 2")
print("W: \n{}".format(W))
print("X: \n{}".format(X))
print("Z: \n{}".format(Z))
