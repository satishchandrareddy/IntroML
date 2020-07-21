# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun == "linear":
		return Z
	elif activation_fun == "sigmoid":
		return 1/(1+np.exp(-Z))

def activation_der(activation_fun,A,grad_A_L):
	if activation_fun == "linear":
		return grad_A_L*np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return grad_A_L*(A - np.square(A))