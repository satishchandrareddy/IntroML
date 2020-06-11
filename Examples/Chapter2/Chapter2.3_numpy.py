# Chapter2.3_2darrays.py

import numpy as np

# Example: 2d arrays
X = np.array([[1,2,3],[4,5,6]])
W = np.array([[1,2,3]])
b = np.array([[4]])
print("X: \n{}".format(X))
print("W: \n{}".format(W))
print("b: \n{}".format(b))

# sum in row direction - this removes an axis -> 1d array
rowsum1 = np.sum(X,axis=0)
print("sum in row direction X: \n{}".format(rowsum1))
print("rowsum1.shape: {}".format(rowsum1.shape))
# sum in row direction - keep row axis -> 2d array
rowsum2 = np.sum(X,axis=0,keepdims=True)
print("sum in row direction X keep row axis: \n{}".format(rowsum2))
print("rowsum2.shape: {}".format(rowsum2.shape))
# sum in column direction - keep column axis -> 2darray
colsum1 = np.sum(X,axis=1,keepdims=True)
print("sum in column direction keep col axis: \n{}".format(colsum1))
print("colsum1.shape: {}".format(colsum1.shape))
# exponential - componentwise exponential
Xexp = np.exp(X)
print("Exponential of X: \n{}".format(Xexp))
# concatenation
Wplusb = np.concatenate((W,b),axis=1)
print("W and b concatenated: {}".format(Wplusb))
# reshape
Xreshape = np.reshape(X,(1,6))
print("X reshape: {}".format(Xreshape))