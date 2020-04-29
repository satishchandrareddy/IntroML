# LRegression.py
#
import functions_activation
import functions_loss
import NeuralNetwork_Base
import numpy as np

class NeuralNetwork(NeuralNetwork_Base.NeuralNetwork_Base):
    def __init__(self,nfeature):
        self.nlayer = 0
        self.nfeature = nfeature
        self.info = []

    def add_layer(self,nunit,activation):
        if self.nlayer == 0:
            nIn = self.nfeature
        else:
            nIn = self.info[self.nlayer-1]["nOut"]
        linfo = {"nIn": nIn, "nOut": nunit, "activation": activation}
        linfo["param"] = {"W": np.random.randn(nunit,nIn), "b": np.random.randn(nunit,1)}
        linfo["param_der"] = {"W": np.zeros((nunit,nIn)), "b": np.zeros((nunit,1))}
        linfo["optimizer"] = {"W": None, "b": None}
        self.info.append(linfo)
        self.nlayer += 1

    def forward_propagate(self,X):
        for layer in range(self.nlayer):
            # linear part
            if layer == 0:
                Ain = X
            else:
                Ain = self.get_A(layer-1)
            Z = np.dot(self.get_param(layer,"param","W"),Ain)+self.get_param(layer,"param","b")
            # activation
            self.info[layer]["A"] = functions_activation.activation(self.info[layer]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute derivative of loss
        dloss_dA = functions_loss.loss_der(self.loss,self.get_A(self.nlayer-1),Y)
        for layer in range(self.nlayer-1,-1,-1):
            # multiply by derivative of A
            dloss_dZ = functions_activation.activation_der(self.info[layer]["activation"],self.get_A(layer),dloss_dA)
            # compute grad_W L and grad_b L
            self.info[layer]["param_der"]["b"] = np.sum(dloss_dZ,axis=1,keepdims=True)
            if layer > 0:
                self.info[layer]["param_der"]["W"] = np.dot(dloss_dZ,self.get_A(layer-1).T)
                dloss_dA = np.dot(self.get_param(layer,"param","W").T,dloss_dZ)
            else:
                self.info[layer]["param_der"]["W"] = np.dot(dloss_dZ,X.T)

    def concatenate_param(self,order):
        flat = np.zeros((1,0))
        for layer in range(self.nlayer):
            row = self.info[layer]["nOut"]
            col = self.info[layer]["nIn"]
            W = np.reshape(self.get_param(layer,order,"W"),(1,self.info[layer]["nIn"]*self.info[layer]["nOut"]))
            b = np.reshape(self.get_param(layer,order,"b"),(1,self.info[layer]["nOut"]))
            flat = np.concatenate((flat,W,b),axis=1)
        return flat

    def load_param(self,flat,order):
        start = 0
        for layer in range(self.nlayer):
            endW = start + self.info[layer]["nIn"]*self.info[layer]["nOut"]
            endb = endW + self.info[layer]["nOut"]
            self.info[layer][order]["W"]=np.reshape(flat[0][start:endW],(self.info[layer]["nOut"],self.info[layer]["nIn"]))
            self.info[layer][order]["b"]=np.reshape(flat[0][endW:endb],(self.info[layer]["nOut"],1))
            start = endb