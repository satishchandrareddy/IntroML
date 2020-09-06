# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	LIM = 50
	if activation_fun == "linear":
		return Z
	elif activation_fun == "sigmoid":
		ZLIM = np.maximum(Z,-LIM)
		return 1/(1+np.exp(-ZLIM))
	elif activation_fun == "mirrorsigmoid":
		ZLIM = np.maximum(Z,LIM)
		return 1/(1+np.exp(ZLIM))
	elif activation_fun == "relu":
		return np.maximum(Z,0.0)
	elif activation_fun == "tanh":
		return np.tanh(Z)
	elif activation_fun == "softplus":
		ZLIM = np.maximum(Z,-LIM)
		return ZLIM + np.log(np.exp(-ZLIM)+1)
	# add elu case
	elif activation_fun == "elu":
		ZLIM = np.maximum(Z,LIM)
		return Z*np.heaviside(Z, 1) + (np.exp(ZLIM)-1)*np.heaviside(-Z, 0)

def activation_der(activation_fun,A,grad_A_L):
	if activation_fun == "linear":
		return grad_A_L*np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return grad_A_L*(A - np.square(A))
	elif activation_fun == "mirrorsigmoid":
		return grad_A_L*(np.square(A) - A)
	elif activation_fun == "relu":
		return grad_A_L*(np.piecewise(A,[A<0,A>=0],[0,1]))
	elif activation_fun == "tanh":
		return grad_A_L*(1 - np.square(A))
	elif activation_fun == "softplus":
		return grad_A_L*(1 - np.exp(-A))
	# add elu case
	elif activation_fun == "elu":
		return grad_A_L*(np.heaviside(A, 1) + (A+1)*np.heaviside(-A, 0))