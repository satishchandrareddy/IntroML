# Chapter1.7_Optimization2.py

import numpy as np
import matplotlib.pyplot as plt

def loss(W):
	return 2*W[0]**2 + W[1]**2

def grad(W):
	return np.array([4*W[0],2*W[1]])

# initialization
W = np.array([2,2])
alpha = 0.1
niter = 30

# iteration
loss_history = []
for iteration in range(niter):
	gradW = grad(W)
	W = W - alpha*gradW
	loss_history.append(loss(W))
print("After {} iterations".format(niter))
print("W: {}".format(W))
print("Loss: {}".format(loss_history[-1]))

plt.figure()
iteration_list = list(range(1,niter+1))
plt.plot(iteration_list,loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()