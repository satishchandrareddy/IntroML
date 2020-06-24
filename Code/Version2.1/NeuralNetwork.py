# NeuralNetwork.py

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
            W = self.get_param(layer,"param","W")
            b = self.get_param(layer,"param","b")
            Z = np.dot(W,Ain)+b
            # activation
            self.info[layer]["A"] = functions_activation.activation(self.info[layer]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute derivative of loss
        grad_A_L = functions_loss.loss_der(self.loss,self.get_A(self.nlayer-1),Y)
        for layer in range(self.nlayer-1,-1,-1):
            # multiply by derivative of A
            grad_Z_L = grad_A_L*functions_activation.activation_der(self.info[layer]["activation"],self.get_A(layer))
            # compute grad_W L and grad_b L
            self.info[layer]["param_der"]["b"] = np.sum(grad_Z_L,axis=1,keepdims=True)
            if layer > 0:
                self.info[layer]["param_der"]["W"] = np.dot(grad_Z_L,self.get_A(layer-1).T)
                grad_A_L = np.dot(self.get_param(layer,"param","W").T,grad_Z_L)
            else:
                self.info[layer]["param_der"]["W"] = np.dot(grad_Z_L,X.T)

    def concatenate_param(self,order):
        # use flat to collect all parameters (initial shape is 1 row and 0 columns)
        flat = np.zeros((1,0))
        for layer in range(self.nlayer):
            # get number rows and columns in W and b in layer
            nrow = self.info[layer]["nOut"]
            ncol = self.info[layer]["nIn"]
            # convert W and b into row vectors and then concatenate to flat
            Wrow = np.reshape(self.get_param(layer,order,"W"),(1,nrow*ncol))
            brow = np.reshape(self.get_param(layer,order,"b"),(1,nrow))
            flat = np.concatenate((flat,Wrow,brow),axis=1)
        return flat

    def load_param(self,flat,order):
        start = 0
        for layer in range(self.nlayer):
        	# get number rows and columns in W and b in layer
            nrow = self.info[layer]["nOut"]
            ncol = self.info[layer]["nIn"]
            # get start and end points of W and b
            endW = start + ncol*nrow
            endb = endW + nrow
            # extract data from flat and put into correct shape into W and b
            self.info[layer][order]["W"]=np.reshape(flat[0][start:endW],(nrow,ncol))
            self.info[layer][order]["b"]=np.reshape(flat[0][endW:endb],(nrow,1))
            start = endb