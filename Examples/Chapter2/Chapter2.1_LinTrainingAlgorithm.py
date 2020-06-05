#Chapter2.1_LinTrainingAlgorithm.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[2,5,7]])
Y = np.array([[8,6,10]])
W = np.array([[1,1]])
b = np.array([[2]])
alpha = 0.01

# ITERATION 1
# forward propagation
Z = np.dot(W,X)+b
A = Z
print("INTERATION 1 ********")
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
# update W and b
W = W - alpha*grad_WL
b = b - alpha*grad_bL
print("W1: {}".format(W))
print("b1: {}".format(b))
# compute loss
Z = np.dot(W,X) + b
A = Z
Loss = np.sum(np.square(A-Y))/3
print("Loss1: {}".format(Loss))


# ITERATiON 2
# forward propagation
Z = np.dot(W,X)+b
A = Z
print("INTERATION 2 ********")
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
# update W and b
W = W - alpha*grad_WL
b = b - alpha*grad_bL
print("W2: {}".format(W))
print("b2: {}".format(b))
# compute loss
Z = np.dot(W,X) + b
A = Z
Loss = np.sum(np.square(A-Y))/3
print("Loss2: {}".format(Loss))