# Chapter1.5_Transpose.py

import numpy as np

# transpose example 1
# W = [1 2 3 4]
W = np.array([[1,2,3,4]])
print("W: \n{}".format(W))
print("W transpose: \n{}".format(W.T))

# transpose example 2
# X = [1  2  3  4]
#     [5  6  7  8]
#     [9 10 11 12]
X = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("X: \n{}".format(X))
print("X transpose: \n{}".format(X.T))