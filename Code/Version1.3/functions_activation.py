# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun == "linear":
		return Z
	elif activation_fun == "sigmoid":
		return 1/(1+np.exp(-Z))

def activation_der(activation_fun,A,dA):
	if activation_fun == "linear":
		return dA*np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return dA*(A - np.square(A))