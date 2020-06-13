# Chapter1.4_NumpyArrays.py

import numpy as np

# Example: 1d-array
# convert list to numpy array
a = np.array([1,2,3])
print("1d array: a: {}".format(a))
print("1d array: a.shape: {}".format(a.shape))

# Example: 2d-array as row vector
# A = [1,2,3]
# Define A as list of list for each row
A = np.array([[1,2,3]])
print("2d array row: A: {}".format(A))
print("2d array row: A.shape: {}".format(A.shape))

# Example: 2d-array as column vector
# each row is a list
# Define B as list of list for each row
# B = [1]
#     [2]
#     [3]
B = np.array([[1],[2],[3]])
print("2d array column: B: {}".format(B))
print("2d array column: B.shape: {}".format(B.shape))

# Example: matrix
# X = [1 2 3]
#     [4 5 6]
# Define X as list of list for each row
X = np.array([[1,2,3],[4,5,6]])
print("2d array: X: {}".format(X))
print("2d array: X.shape: {}".format(X.shape))