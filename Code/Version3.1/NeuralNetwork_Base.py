# NeuralNetwork_Base.py

from copy import deepcopy
import numpy as np
import functions_loss
import onehot
import Optimizer

class NeuralNetwork_Base:
    def __init__(self):
        pass 

    def compile(self,loss_fun,optimizer_object):
        self.loss = loss_fun
        # assign deepcopy of optimizer object to W and b for each layer
        for layer in range(self.nlayer):
            self.info[layer]["optimizer"]["W"] = deepcopy(optimizer_object)
            self.info[layer]["optimizer"]["b"] = deepcopy(optimizer_object)

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

    def update_param(self):
        # Update the parameter matrices W and b for each layer in neural network
        for layer in range(self.nlayer):
            # paramter_guess= i = parameter_guess=i-1 + update_guess=i-1
            self.info[layer]["param"]["W"] += self.info[layer]["optimizer"]["W"].update(self.get_param(layer,"param_der","W"))
            self.info[layer]["param"]["b"] += self.info[layer]["optimizer"]["b"].update(self.get_param(layer,"param_der","b"))

    def fit(self,X,Y,epochs,**kwargs):
        # iterate over epochs
        loss = []
        accuracy = []
        # get mini-batches
        if "batch_size" in kwargs:
            mini_batch = self.mini_batch(X,Y,kwargs["batch_size"])
        else:
            mini_batch = [(X,Y)]
        # train
        for epoch in range(epochs):
            # train using mini-batches
            for (Xbatch,Ybatch) in mini_batch:
                self.forward_propagate(Xbatch)
                self.back_propagate(Xbatch,Ybatch)
                self.update_param()
            # compute loss and accuracy after cycling through all mini-batches    
            Y_pred = self.predict(X)
            loss.append(self.compute_loss(Y))
            accuracy.append(self.accuracy(Y,Y_pred))
            print("Epoch: {} - Cost: {} - Accuracy: {}".format(epoch+1,loss[epoch],accuracy[epoch]))
        return {"loss":loss,"accuracy":accuracy}

    def predict(self,X):
        self.forward_propagate(X)
        if self.info[self.nlayer-1]["activation"]=="linear":
            return self.get_A(self.nlayer-1)
        elif self.info[self.nlayer-1]["activation"]=="sigmoid":
            return np.round(self.get_A(self.nlayer-1),0)
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
            print("{}\t{}\t\t{}\t\t{}".format(layer+1,self.info[layer]["nIn"],self.info[layer]["nOut"],nparameter))
        print
        print("Total parameters: {}".format(nparameter_total))
        print(" ")

    def mini_batch(self,X,Y,batch_size):
        m = Y.shape[1]
        # determine number of mini-batches
        if m % batch_size == 0:
            n = int(m/batch_size)
        else:
            n = int(m/batch_size) + 1
        # create mini-batches
        mini_batch = []
        for count in range(n):
            start = count*batch_size
            end = start + min(start+batch_size,m)
            mini_batch.append((X[:,start:end],Y[:,start:end]))
        return mini_batch