# Chapter3.1_TrainingAlgorithm.py

import numpy as np

W1 = np.array([[0.5,0.5],[0.5,-0.5]])
b1 = np.array([[0.5],[0.5]])
W2 = np.array([[-1,1]])
b2 = np.array([[-0.1]])
grad_W1L = np.array([[-0.3967, 0.7711],[0.0165,-0.0328]])
grad_b1L = np.array([[-0.2639],[0.0165]])
grad_W2L = np.array([[-0.2186, 0.4591]])
grad_b2L = np.array([[0.4675]])
alpha = 0.1
print("W1 update: {}".format(W1 - alpha*grad_W1L))
print("b1 update: {}".format(b1 - alpha*grad_b1L))
print("W2 update: {}".format(W2 - alpha*grad_W2L))
print("b2 update: {}".format(b2 - alpha*grad_b2L))