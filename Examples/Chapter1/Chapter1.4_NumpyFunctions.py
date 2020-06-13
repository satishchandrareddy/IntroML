# Chapter1.4_NumpyFunctions.py

import numpy as np

# X = [1 2 3]
#     [4 5 6]
# Y = [1 -1  2]
#     [2  3 -2]
X = np.array([[1,2,3],[4,5,6]])
Y = np.array([[1,-1,2],[2,3,-2]])
print("X: \n{}".format(X))
print("Y: \n{}".format(Y))
# Numpy functions operate on each entry separately
# exponential
Xexp = np.exp(X)
print("Exponential of X: \n{}".format(Xexp))
# absolute value
Yabs = np.absolute(Y)
print("Absolute Value of Y: \n{}".format(Yabs))
# square
Xsq = np.square(X)
print("Square of X: \n{}".format(Xsq))