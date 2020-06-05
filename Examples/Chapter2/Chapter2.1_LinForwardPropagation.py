#Chapter2.1_LinForwardPropagation.py

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