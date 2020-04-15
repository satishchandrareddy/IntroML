# functions_loss.py
#
import numpy as np

def loss(loss_fun,A,Y):
	if loss_fun == "meansquarederror":
		return np.mean(np.square(A-Y))
	elif loss_fun == "binarycrossentropy":
		return -np.mean(Y*np.log(A)+(1-Y)*np.log(1-A))

def loss_der(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return 2*(A-Y)/m
	elif loss_fun == "binarycrossentropy":
		return (-Y/(A+1e-16) + (1-Y)/(1-A+1e-16))/m