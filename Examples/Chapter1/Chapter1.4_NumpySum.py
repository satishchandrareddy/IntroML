# Chapter1.4_NumpySum.py

import numpy as np

# X = [1 2 3]
#     [4 5 6]
X = np.array([[1,2,3],[4,5,6]])
print("X: \n{}".format(X))
# sum all entries
sum1 = np.sum(X)
print("sum entries of x: {}".format(sum1))
# sum in row direction - this removes an axis -> 1d array
# [5 7 9]
rowsum1 = np.sum(X,axis=0)
print("sum in row direction X: \n{}".format(rowsum1))
print("rowsum1.shape: {}".format(rowsum1.shape))
# sum in row direction - keep row axis -> 2d array
# [5 7 9]
rowsum2 = np.sum(X,axis=0,keepdims=True)
print("sum in row direction X keep row axis: \n{}".format(rowsum2))
print("rowsum2.shape: {}".format(rowsum2.shape))
# sum in column direction - keep column axis -> 2darray
colsum1 = np.sum(X,axis=1,keepdims=True)
print("sum in column direction keep col axis: \n{}".format(colsum1))
print("colsum1.shape: {}".format(colsum1.shape))