# Chapter1.4_NumpyShapes.py

import numpy as np

# X = [1 2 3]
#     [4 5 6]
# Y = [1 -1  2]
#     [2  3 -2]
X = np.array([[1,2,3],[4,5,6]])
Y = np.array([[1,-1,2],[2,3,-2]])
print("X: \n{}".format(X))
print("Y: \n{}".format(Y))
# concatenation - in row direction
# result is [1  2  3]
#           [4  5  6]
#           [1 -1  2]
#           [2  3 -2]
XandYrow = np.concatenate((X,Y),axis=0)
print("X and Y concatenated in row direction: \n{}".format(XandYrow))
# concatenation X and Y - in column direction
# result is [1 2 3 1 -1  2]
#           [4 5 6 2  3 -2]
XandYcol = np.concatenate((X,Y),axis=1)
print("X and Y concatenated in column direction: \n{}".format(XandYcol))
# reshape X into matrix 1 row 6 columns: [1 2 3 4 5 6]
Xreshape = np.reshape(X,(1,6))
print("X reshape: \n{}".format(Xreshape))