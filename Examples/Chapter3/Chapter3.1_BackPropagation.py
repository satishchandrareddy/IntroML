#Chapter3.1_LogBackPropagation.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[-2,-5,-8]])
Y = np.array([[0,1,0]])
W1 = np.array([[0.5,0.5],[0.5,-0.5]])
b1 = np.array([[0.5],[0.5]])
W2 = np.array([[-1,1]])
b2 = np.array([[-0.1]])
# forward propagation
# layer 1
Z1 = np.dot(W1,X)+b1
A1 = np.tanh(Z1)
print("Z1: {}".format(Z1))
print("A1: {}".format(A1))
# layer2
Z2 = np.dot(W2,A1)+b2
A2 = 1/(1+np.exp(-Z2))
print("Z2: {}".format(Z2))
print("A2: {}".format(A2))
# back propagation
dLdA2 = -1/3*(Y/A2 - (1-Y)/(1-A2))
print("dLdA2: {}".format(dLdA2))
# layer 2
dA2dZ2 = A2 - np.square(A2)
print("dA2dZ2: {}".format(dA2dZ2))
dLdZ2 = dLdA2*dA2dZ2
print("dLdZ2: {}".format(dLdZ2))
grad_W2L = np.dot(dLdZ2,A1.T)
grad_b2L = np.sum(dLdZ2,axis=1,keepdims=True)
print("grad_W2L: {}".format(grad_W2L))
print("grad_b2L: {}".format(grad_b2L))
# layer 1
dLdA1 = np.dot(W2.T,dLdZ2)
print("dLdA1: {}".format(dLdA1))
dA1dZ1 = 1 - np.square(A1)
print("dA1dZ1: {}".format(dA1dZ1))
dLdZ1 = dLdA1*dA1dZ1
print("dLdZ1: {}".format(dLdZ1))
grad_W1L = np.dot(dLdZ1,X.T)
grad_b1L = np.sum(dLdZ1,axis=1,keepdims=True)
print("grad_W1L: {}".format(grad_W1L))
print("grad_b1L: {}".format(grad_b1L))