# functions_loss.py

import numpy as np

def loss(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return np.sum(np.square(A-Y))/m
	# add logcosh case
	elif loss_fun == "logcosh":
		return np.sum(np.log(np.cosh(A-Y)))/m

def loss_der(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return 2*(A-Y)/m
	# add logcosh gradient
	elif loss_fun == "logcosh":
		return np.tanh(A-Y)/m