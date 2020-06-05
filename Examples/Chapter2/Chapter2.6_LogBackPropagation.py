#Chapter2.6_LogBackPropagation.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[-2,-5,-8]])
Y = np.array([[0,1,0]])
W = np.array([[0.1,0.1]])
b = np.array([[0.2]])
# forward propagation
Z = np.dot(W,X)+b
A = 1/(1+np.exp(-Z))
print("Z: {}".format(Z))
print("A: {}".format(A))
# back propagation
grad_AL = -1/3*(Y/A - (1-Y)/(1-A))
print("grad_AL: {}".format(grad_AL))
dAdZ = A-np.square(A)
print("dAdZ: {}".format(dAdZ))
grad_ZL = grad_AL*dAdZ
print("grad_ZL: {}".format(grad_ZL))
grad_WL = np.dot(grad_ZL,X.T)
grad_bL = np.sum(grad_ZL,axis=1,keepdims=True)
print("grad_WL: {}".format(grad_WL))
print("grad_bL: {}".format(grad_bL))