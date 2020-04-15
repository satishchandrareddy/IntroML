# functions_loss.py
#
import numpy as np

def loss(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return np.sum(np.square(A-Y))/m

def loss_der(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return 2*(A-Y)/m