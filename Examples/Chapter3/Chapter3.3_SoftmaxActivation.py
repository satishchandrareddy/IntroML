# Chapter3.3_SoftmaxActivation.py

import numpy as np

Z = np.array([[0.1,-0.1,-0.2],[-0.2, 0.2, 0.3],[-0.3,0.1,0.2],[0.4,-0.3,-0.5]])
Zexp = np.exp(Z)
A = Zexp/np.sum(Zexp,axis=0)
print("Z: {}".format(Z))
print("Z.shape: {}".format(Z.shape))
print("A: {}".format(A))
print("A.shape: {}".format(A.shape))
# sum along each column - should be 1
print("Sum: {}".format(np.sum(A,axis=0)))