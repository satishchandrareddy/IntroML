# LRegression.py

import functions_activation
import functions_loss
import NeuralNetwork_Base
import numpy as np

class LRegression(NeuralNetwork_Base.NeuralNetwork_Base):
    def __init__(self,nfeature,activation):
        self.nlayer = 1
        self.info = [{"nIn": nfeature, "nOut": 1, "activation": activation}]
        self.info[0]["param"] = {"W": np.random.randn(1,self.info[0]["nIn"]), "b": np.random.randn(1,1)}
        self.info[0]["param_der"] = {"W": np.zeros((1,self.info[0]["nIn"])), "b": np.zeros((1,1))}
        self.info[0]["optimizer"] = {"W": None, "b": None}

    def forward_propagate(self,X):
        W = self.get_param(0,"param","W")
        b = self.get_param(0,"param","b")
        # linear part
        Z = np.dot(W,X) + b
        # activation
        self.info[0]["A"] = functions_activation.activation(self.info[0]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute gradient of loss with respect to A
        grad_A_L = functions_loss.loss_der(self.loss,self.get_A(0),Y)
        # compute gradient of loss with respect to Z
        grad_Z_L = functions_activation.activation_der(self.info[0]["activation"],self.get_A(0),grad_A_L)
        # compute grad_W L and grad_b L
        self.info[0]["param_der"]["b"] = np.sum(grad_Z_L,axis=1,keepdims=True)
        self.info[0]["param_der"]["W"] = np.dot(grad_Z_L,X.T)

    def concatenate_param(self,order):
        # concatenate W and b or (grad W and grad b) into single row 
        return np.concatenate((self.get_param(0,order,"W"),self.get_param(0,order,"b")),axis=1)

    def load_param(self,flat,order):
        ncol = self.info[0]["nIn"]
        # W consists of the first set of entries
        self.info[0][order]["W"]=flat[:,0:ncol]
        # b is the final entry
        self.info[0][order]["b"]=flat[:,ncol:ncol+1]