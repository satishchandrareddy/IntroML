# LRegression.py
#
import functions_activation
import functions_loss
import NeuralNetwork
import numpy as np

class LRegression(NeuralNetwork.NeuralNetwork_Base):
    def __init__(self,n_features,activation):
        self.n_layer = 1
        self.info = [{"n_In": n_features, "activation": activation}]
        self.info[0]["params"] = {"W": np.random.randn(1,self.info[0]["n_In"]), "b": np.random.randn(1,1)}
        self.info[0]["params_der"] = {"W": np.zeros((1,self.info[0]["n_In"])), "b": np.zeros((1,1))}

    def forward_propagate(self,X):
        Z = np.dot(self.get_params(0,"params","W"),X) + self.get_params(0,"params","b")
        self.info[0]["A"] = functions_activation.activation(self.info[0]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute derivative of loss
        dloss_dA = functions_loss.loss_der(self.loss,self.get_A(0),Y)
        # multiply by derivative of A
        dloss_dZ = dloss_dA*functions_activation.activation_der(self.info[0]["activation"],self.get_A(0))
        # compute grad_W L and grad_b L
        self.info[0]["params_der"]["b"] = np.sum(dloss_dZ,axis=1,keepdims=True)
        self.info[0]["params_der"]["W"] = np.dot(dloss_dZ,X.T)

    def concatenate_parameters(self,order):
        flat = np.concatenate((self.get_params(0,order,"W"),self.get_params(0,order,"b")),axis=1)
        return flat

    def load_parameters(self,flat,order):
        ncol = self.info[0]["n_In"]
        self.info[0][order]["W"]=flat[:,0:ncol]
        self.info[0][order]["b"]=flat[:,ncol:ncol+1]