# NeuralNetwork_Base.py

import functions_loss
import numpy as np
import onehot
import Optimizer

class NeuralNetwork_Base:
    def __init__(self):
        pass 

    def compile(self,loss_fun,dict_opt):
        self.loss = loss_fun
        for layer in range(self.nlayer):
            for label in self.get_param_label(layer):
                self.info[layer]["optimizer"][label] = Optimizer.constructor(dict_opt)

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

    def compute_loss(self,Y):
        return functions_loss.loss(self.loss,self.get_A(self.nlayer-1),Y)

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

    def get_param_label(self,layer):
        return self.info[layer]["param"].keys()

    def update_param(self):
        for layer in range(self.nlayer):
            for label in self.get_param_label(layer):
                self.info[layer]["param"][label] += self.info[layer]["optimizer"][label].update(self.get_param(layer,"param_der",label))

    def train(self,X,Y,epochs):
        # iterate over epochs
        loss_history = []
        accuracy_history = []
        for epoch in range(epochs):
            self.forward_propagate(X)
            self.back_propagate(X,Y)
            self.update_param()
            Y_pred = self.predict(X)
            loss_history.append(self.compute_loss(Y))
            accuracy_history.append(self.accuracy(Y,Y_pred))
            print("Epoch: {} - Cost: {} - Accuracy: {}".format(epoch,loss_history[epoch],accuracy_history[epoch]))
        return {"loss":np.array(loss_history),"accuracy":np.array(accuracy_history)}

    def predict(self,X):
        self.forward_propagate(X)
        if self.info[self.nlayer-1]["activation"]=="sigmoid":
            return np.round(self.get_A(self.nlayer-1),0)
        elif self.info[self.nlayer-1]["activation"]=="linear":
            return self.get_A(self.nlayer-1)
        elif self.info[self.nlayer-1]["activation"]=="softmax":
            return np.expand_dims(np.argmax(self.get_A(self.nlayer-1),0),axis=0)

    def accuracy(self,Y,Y_pred):
        if self.loss == "meansquarederror":
            return np.mean(np.absolute(Y - Y_pred))
        elif self.loss == "binarycrossentropy":
            return np.mean(np.absolute(Y-Y_pred)<1e-7)
        elif self.loss == "crossentropy":
            return np.mean(np.absolute(Y-Y_pred)<1e-7)
           
    def summary(self):
        print(" ")
        print("Layer\tUnits In\tUnits Out\tParameters")
        nparameter_total = 0
        for layer in range(self.nlayer):
            nparameter = (self.info[layer]["nIn"]+1)*self.info[layer]["nOut"]
            nparameter_total += nparameter
            print("{}\t{}\t\t{}\t\t{}".format(layer,self.info[layer]["nIn"],self.info[layer]["nOut"],nparameter))
        print
        print("Total parameters: {}".format(nparameter_total))
        print(" ")

    