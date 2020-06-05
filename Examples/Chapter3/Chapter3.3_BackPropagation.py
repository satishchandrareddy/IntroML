# Chapter3.3_BackPropagation.py
#
import numpy as np

def onehot(Y,nclass):
    ndata = Y.shape[1]
    Y_onehot = np.zeros((nclass,ndata))
    for count in range(ndata):
        Y_onehot[int(Y[0,count]),count] = 1.0
    return Y_onehot

# training data
X = np.array([[1,2,4],[-2,-5,-8]])
Y = np.array([[0,1,2]])
# parameters
W1 = np.array([[0.5,0.5],[0.5,-0.5]])
b1 = np.array([[0.5],[0.5]])
W2 = np.array([[-1,1],[1,-1],[-2,1]])
b2 = np.array([[-0.1]])
print("FORWARD PROPAGATION")
# layer 1
Z1 = np.dot(W1,X) + b1
print("Z1: {}".format(Z1))
A1 = np.tanh(Z1)
print("A1: {}".format(A1))
# layer 2
Z2 = np.dot(W2,A1) + b2
print("Z2: {}".format(Z2))
Z2exp = np.exp(Z2)
A2 = Z2exp/np.sum(Z2exp,axis=0)
print("A2: {}".format(A2))
#
print("BACK PROPAGATION")
# dLoss/dA2
Yonehot = onehot(Y,3)
print("Yonehot: {}".format(Yonehot))
dLossdA2 = -Yonehot/A2/3
print("dLossdA2: {}".format(dLossdA2))
# LAYER 2
# dLoss/dZ2
prod2 = A2*dLossdA2
print("A2*dLossdA: {}".format(prod2))
sumterm = np.sum(prod2,axis=0)
print("sumterm: {}".format(sumterm))
sumprod = A2*sumterm
print("sumprod: {}".format(sumprod))
dLossdZ2 = prod2 - A2*sumprod
print("dLossdZ2: {}".format(dLossdZ2))
dLossdW2 = np.dot(dLossdZ2,A1.T)
dLossdb2 = np.sum(dLossdZ2,axis=1)
dLossdA1 = np.dot(W2.T,dLossdZ2)
print("dLossdW2: {}".format(dLossdW2))
print("dLossdb2: {}".format(dLossdb2))
print("dLossdA1: {}".format(dLossdA1))
# LAYER 1
dA1dZ1 = 1-A1*A1
print("dA1/dZ1: {}".format(dA1dZ1))
dLossdZ1 = dLossdA1*dA1dZ1
print("dLossdZ1: {}".format(dLossdZ1))
dLossdW1 = np.dot(dLossdZ1,X.T)
dLossdb1 = np.sum(dLossdZ1,axis=1,keepdims=True)
print("dLossdW1: {}".format(dLossdW1))
print("dLossdb1: {}".format(dLossdb1))