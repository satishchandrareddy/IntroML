#Chapter2.1_LinBackPropagation.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[2,5,7]])
Y = np.array([[8,6,10]])
W = np.array([[1,1]])
b = np.array([[2]])
# forward propagation
Z = np.dot(W,X)+b
A = Z
print("Z: {}".format(Z))
print("A: {}".format(A))
# back propagation
grad_AL = 2/3*(A-Y)
print("grad_AL: {}".format(grad_AL))
dAdZ = np.ones(Y.shape)
print("dAdZ: {}".format(dAdZ))
grad_ZL = grad_AL*dAdZ
print("grad_ZL: {}".format(grad_ZL))
grad_WL = np.dot(grad_ZL,X.T)
grad_bL = np.sum(grad_ZL,axis=1,keepdims=True)
print("grad_WL: {}".format(grad_WL))
print("grad_bL: {}".format(grad_bL))