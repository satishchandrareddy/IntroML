# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun=="linear":
		return Z

def activation_der(activation_fun,A,grad_A_L):
	if activation_fun == "linear":
		return grad_A_L*np.ones(A.shape)