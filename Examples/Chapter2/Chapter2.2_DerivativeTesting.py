#Chapter2.2_DerivativeTesting.py

import numpy as np 

def Loss(A,Y):
	return np.sum(np.square(A-Y))/3

# inputs
X = np.array([[1,2,4],[2,5,7]])
Y = np.array([[8,6,10]])
W = np.array([[1,1]])
b = np.array([[2]])
grad_WL = np.array([[10,20]])
grad_bL = np.array([[2]])
eps = 0.1
# estimated dLdW0
print("dLdW0 ****")
W = np.array([[1+eps,1]])
Z = np.dot(W,X) + b
A = Z
print("A plus: {}".format(A))
Lossp = Loss(A,Y)
print("Loss plus: {}".format(Lossp))
W = np.array([[1-eps,1]])
Z = np.dot(W,X) + b
A = Z
print("A minus: {}".format(A))
Lossm = Loss(A,Y)
print("Loss minus: {}".format(Lossm))
dLdW0 = (Lossp - Lossm)/2/eps
print("Estimated dL/dW0: {}".format(dLdW0))
# estimated dLdW1
print("dLdW1 ****")
W = np.array([[1,1+eps]])
Z = np.dot(W,X) + b
A = Z
print("A plus: {}".format(A))
Lossp = Loss(A,Y)
print("Loss plus: {}".format(Lossp))
W = np.array([[1,1-eps]])
Z = np.dot(W,X) + b
A = Z
print("A minus: {}".format(A))
Lossm = Loss(A,Y)
print("Loss minus: {}".format(Lossm))
dLdW1 = (Lossp - Lossm)/2/eps
print("Estimated dL/dW1: {}".format(dLdW1))
# estimated dLdb
print("dLdb ****")
W = np.array([[1,1]])
b = np.array([[2+eps]])
Z = np.dot(W,X) + b
A = Z
print("A plus: {}".format(A))
Lossp = Loss(A,Y)
print("Loss plus: {}".format(Lossp))
b = np.array([[2-eps]])
Z = np.dot(W,X) + b
A = Z
print("A minus: {}".format(A))
Lossm = Loss(A,Y)
print("Loss minus: {}".format(Lossm))
dLdb = (Lossp - Lossm)/2/eps
print("Estimated dL/db: {}".format(dLdb))