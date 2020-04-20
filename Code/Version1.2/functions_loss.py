# functions_loss.py

import numpy as np

def loss(loss_fun,A,Y):
	if loss_fun == "meansquarederror":
		return np.mean(np.square(A-Y))

def loss_der(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return 2*(A-Y)/m