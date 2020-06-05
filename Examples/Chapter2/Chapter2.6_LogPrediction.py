#Chapter2.6_LogPrediction.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[-2,-5,-8]])
Y = np.array([[0,1,0]])
W = np.array([[0.1,0.1]])
b = np.array([[0.2]])
# forward propagation
Z = np.dot(W,X)+b
A = 1/(1+np.exp(-Z))
print("A: {}".format(A))
print("round A: {}".format(np.round(A)))