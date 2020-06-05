#Chapter3.1_LogForwardPropagation.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[-2,-5,-8]])
Y = np.array([[0,1,0]])
W1 = np.array([[0.5,0.5],[0.5,-0.5]])
b1 = np.array([[0.5],[0.5]])
W2 = np.array([[-1,1]])
b2 = np.array([[-0.1]])
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