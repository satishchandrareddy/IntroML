# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun=="linear":
		return Z

def activation_der(activation_fun,A,dA):
	if activation_fun == "linear":
		return dA*np.ones(A.shape)