# Chapter3.3_forwardbackpropagation.py
#
import numpy as np

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