# onehot.py

import numpy as np

def onehot(Y,nclass):
    ndata = Y.shape[1]
    Y_onehot = np.zeros((nclass,ndata))
    for count in range(ndata):
        Y_onehot[int(Y[0,count]),count] = 1.0
    return Y_onehot

def onehot_inverse(Y_onehot):
    ndata = Y_onehot.shape[1]
    return np.reshape(np.argmax(Y_onehot,axis=0),(1,ndata))