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

    def forward_propagate(self,X):
        Z = np.dot(self.get_param(0,"param","W"),X) + self.get_param(0,"param","b")
        self.info[0]["A"] = functions_activation.activation(self.info[0]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute derivative of loss
        dloss_dA = functions_loss.loss_der(self.loss,self.get_A(0),Y)
        # multiply by derivative of A
        dloss_dZ = dloss_dA*functions_activation.activation_der(self.info[0]["activation"],self.get_A(0))
        # compute grad_W L and grad_b L
        self.info[0]["param_der"]["b"] = np.sum(dloss_dZ,axis=1,keepdims=True)
        self.info[0]["param_der"]["W"] = np.dot(dloss_dZ,X.T)

    def concatenate_param(self,order):
        # concatenate W and b or (grad W and grad b) into single row 
        return np.concatenate((self.get_param(0,order,"W"),self.get_param(0,order,"b")),axis=1)

    def load_param(self,flat,order):
        ncol = self.info[0]["nIn"]
        self.info[0][order]["W"]=flat[:,0:ncol]
        self.info[0][order]["b"]=flat[:,ncol:ncol+1]