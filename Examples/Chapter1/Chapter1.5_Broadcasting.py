# Chapter1.5_Broadcasting.py

import numpy as np

# broadcastng example 1
# W = [1 2 3]
# X = [4 7]
#     [5 8]
#     [6 9]
W = np.array([[1,2,3]])
X = np.array([[4,7],[5,8],[6,9]])
b = 7
Z = np.dot(W,X) + b
print("Broadcasting Example 1")
print("W: \n{}".format(W))
print("X: \n{}".format(X))
print("b: \n{}".format(b))
print("Z: \n{}".format(Z))

# broadcasting example 2
# W = [1 2 3]
#     [2 3 4]
# X = [4 7]
#     [5 8]
#     [6 9]
# b = [11]
#     [12]
W = np.array([[1,2,3],[2,3,4]])
X = np.array([[4,7],[5,8],[6,9]])
b = np.array([[11],[12]])
Z = np.dot(W,X) + b
print("Broadcasting Example 2")
print("W: \n{}".format(W))
print("X: \n{}".format(X))
print("b: \n{}".format(b))
print("Z: \n{}".format(Z))