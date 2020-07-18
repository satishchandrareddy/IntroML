# NeuralNetwork_Base.py

import numpy as np
import functions_loss

class NeuralNetwork_Base:
    def __init__(self):
        pass 

    def compile(self,loss_fun,optimizer):
        self.loss = loss_fun
        self.optimizer = optimizer

    def forward_propagate(self,X):
        pass

    def back_propagate(self,X,Y):
        pass

    def get_param(self,layer,order,label):
        # layer = integer: layer number 
        # order = string: "param" or "param_der"
        # label = string: "W" or "b"
        return self.info[layer][order][label]

    def get_A(self,layer):
        return self.info[layer]["A"]

    def get_Afinal(self):
        return self.info[self.nlayer-1]["A"]

    def compute_loss(self,Y):
        return functions_loss.loss(self.loss,self.get_Afinal(),Y)

    def test_derivative(self,X,Y,eps):
        # compute gradients
        self.forward_propagate(X)
        self.back_propagate(X,Y)
        # concatenate parameters and gradients
        param_original = self.concatenate_param("param")
        param_der_model = self.concatenate_param("param_der")
        #approximate derivative using centred-differences bump of eps
        nparam = param_original.shape[1]
        param_der_approx = np.zeros((1,nparam))
        for idx in range(nparam):
            # cost plus
            param_plus = param_original.copy()
            param_plus[:,idx] += eps
            self.load_param(param_plus,"param")
            self.forward_propagate(X)
            cost_plus = self.compute_loss(Y)
            # cost minus
            param_minus = param_original.copy()
            param_minus[:,idx] -= eps
            self.load_param(param_minus,"param")
            self.forward_propagate(X)
            cost_minus = self.compute_loss(Y)
            # apply centred difference formula
            param_der_approx[:,idx] = (cost_plus - cost_minus)/(2*eps)
        # estimate accuracy of derivatives
        abs_error = np.absolute(param_der_model - param_der_approx)
        rel_error = abs_error/(np.absolute(param_der_approx)+1e-15)
        error = min(np.max(abs_error),np.max(rel_error))
        return error