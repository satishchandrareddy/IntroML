# Chapter2.3_2darrays.py

import numpy as np

# Example: 1d-array
a = np.array([1,2,3])
print("1d array: a: {}".format(a))
print("1d array: a.shape: {}".format(a.shape))

# Example: 2d-array as row vector
A = np.array([[1,2,3]])
print("2d array row: A: {}".format(A))
print("2d array row: A.shape: {}".format(A.shape))

# Example: 2d-array as column vector
B = np.array([[1],[2],[3]])
print("2d array column: B: {}".format(B))
print("2d array column: B.shape: {}".format(B.shape))
