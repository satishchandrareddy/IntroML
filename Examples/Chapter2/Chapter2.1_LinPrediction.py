#Chapter2.1_LinPrediction.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[2,5,7]])
Y = np.array([[8,6,10]])
W = np.array([[0.8683,0.7325]])
b = np.array([[1.9837]])
# forward propagation
Z = np.dot(W,X)+b
A = Z
print("Z: {}".format(Z))
print("A: {}".format(A))