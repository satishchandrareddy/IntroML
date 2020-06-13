# Chapter1.4_NumpyComponentWise.py

import numpy as np

# X = [1 2 3]
#     [4 5 6]
# Y = [1 -1  2]
#     [2  3 -2]
X = np.array([[1,2,3],[4,5,6]])
Y = np.array([[1,-1,2],[2,3,-2]])
print("X: \n{}".format(X))
print("Y: \n{}".format(Y))
# Add X and Y
Z1 = X + Y
print("Z1=X+Y: \n{}".format(Z1))
# scalar multiplication
Z2 = 2*X
print("Z2=2*X: \n{}".format(Z2))
# componentwise multiplication
Z3 = X*Y
print("Z3=X*Y: \n{}".format(Z3))