# Chapter4.3_MiniBatchOptimization

import numpy as np

# batches
X = np.array([[1, 2, 4, 1, -1],[-2,-5,-8,1,1]])
Y = np.array([[0, 1, 0, 1, 1]])

# initial parameters
W = np.array([[0.1,0.1]])
b = np.array([[0.2]])

# learning rate
alpha = 0.1

# Epoch 1, Mini-batch sample 0
# Forward propagation
X0 = X[:,0:1]
Y0 = Y[:,0:1]
Z0 = np.dot(W,X0) + b
A0 = 1/(1+np.exp(-Z0))
print("Forward Propagation: Epoch 1, Sample 0")
print("Z0: {}".format(Z0))
print("A0: {}".format(A0))

# Back Propagation
dLossdA0 = -(Y0/A0 - (1-Y0)/(1-A0))/1
dA0dZ0 = A0 - np.square(A0)
dLossdZ0 = dLossdA0*dA0dZ0
dgradW0 = np.dot(dLossdZ0,X0.T)
dgradb0 = np.sum(dLossdZ0,keepdims=True)
print("Back Propagation: Batch0")
print("dLossdA0: {}".format(dLossdA0))
print("dA0dZ0: {}".format(dA0dZ0))
print("dLossdZ0: {}".format(dLossdZ0))
print("dgradW: {}".format(dgradW0))
print("dgradb: {}".format(dgradb0))

# Update W and b
W = W - alpha*dgradW0
b = b - alpha*dgradb0
print("W: {}".format(W))
print("b: {}".format(b))

# Epoch 1, Sample 1
# Forward propagation
X1 = X[:,1:2]
Y1 = Y[:,1:2]
Z1 = np.dot(W,X1) + b
A1 = 1/(1+np.exp(-Z1))
print("Forward Propagation: Epoch 1, Sample 1")
print("Z1: {}".format(Z1))
print("A1: {}".format(A1))

# Back Propagation
dLossdA1 = -(Y1/A1 - (1-Y1)/(1-A1))/1
dA1dZ1 = A1 - np.square(A1)
dLossdZ1 = dLossdA1*dA1dZ1
dgradW1 = np.dot(dLossdZ1,X1.T)
dgradb1 = np.sum(dLossdZ1,keepdims=True)
print("Back Propagation: Batch2")
print("dLossdA1: {}".format(dLossdA1))
print("dA1dZ1: {}".format(dA1dZ1))
print("dLossdZ1: {}".format(dLossdZ1))
print("dgradW: {}".format(dgradW1))
print("dgradb: {}".format(dgradb1))
# Update W and b
W = W - alpha*dgradW1
b = b - alpha*dgradb1
print("W: {}".format(W))
print("b: {}".format(b))
