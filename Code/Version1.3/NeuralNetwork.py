# NeuralNetwork.py
#
# NeuralNetwork Base class
import numpy as np
import functions_loss

class NeuralNetwork_Base:
    def __init__(self,n_features):
        pass 

    def compile(self,optimizer,loss_fun):
        self.optimizer = optimizer
        self.loss = loss_fun

    def forward_propagate(self,X):
        pass

    def back_propagate(self,X,Y):
        pass

    def get_params(self,layer,order,label):
        # layer = integer: layer number 
        # order = string: "params" or "params_der"
        # label = string: "W" or "b"
        return self.info[layer][order][label]

    def get_A(self,layer):
        return self.info[layer]["A"]

    def compute_loss(self,Y):
        return functions_loss.loss(self.loss,self.get_A(self.n_layer-1),Y)

    def test_derivative(self,X,Y,eps):
        # compute derivatives
        self.forward_propagate(X)
        self.back_propagate(X,Y)
        # concatenate
        params_original = self.concatenate_parameters("params")
        params_der_model = self.concatenate_parameters("params_der")
        #approximate derivative using difference centred-differences bump of eps
        n_params = params_original.shape[1]
        params_der_approx = np.zeros((1,n_params))
        for idx in range(n_params):
            # cost plus
            params_plus = params_original.copy()
            params_plus[:,idx] += eps
            self.load_parameters(params_plus,"params")
            self.forward_propagate(X)
            cost_plus = self.compute_loss(Y)
            # cost minus
            params_minus = params_original.copy()
            params_minus[:,idx] -= eps
            self.load_parameters(params_minus,"params")
            self.forward_propagate(X)
            cost_minus = self.compute_loss(Y)
            # apply centred difference formula
            params_der_approx[:,idx] = (cost_plus - cost_minus)/(2*eps)
        # estimate accuracy of derivatives
        abs_error = np.absolute(params_der_model - params_der_approx)
        rel_error = abs_error/(np.absolute(params_der_approx)+1e-15)
        error = min(np.max(abs_error),np.max(rel_error))
        return error

    def update_params(self,layer,label,update):
        self.info[layer]["params"][label] += update  

    def train(self,X,Y,epoch):
        loss_history = []
        accuracy_history = []
        # iterate over epochs
        for iter in range(1,epoch+1):
            self.forward_propagate(X)
            self.back_propagate(X,Y)
            self.optimizer.update_params(self)
            Y_pred = self.predict(X)
            loss_history.append(self.compute_loss(Y))
            accuracy_history.append(self.accuracy(Y,Y_pred))
            print("Epoch: {} - Cost: {} - Accuracy: {}".format(iter,loss_history[iter-1],accuracy_history[iter-1]))
        return {"loss":np.array(loss_history),"accuracy":np.array(accuracy_history)}

    def predict(self,X):
        self.forward_propagate(X)
        if self.info[self.n_layer-1]["activation"]=="sigmoid":
            return np.round(self.get_A(self.n_layer-1),0)
        elif self.info[self.n_layer-1]["activation"]=="linear":
            return self.get_A(self.n_layer-1)

    def accuracy(self,Y,Y_pred):
        if self.loss == "meansquarederror":
            return np.mean(np.absolute(Y - Y_pred))
        elif self.loss == "binarycrossentropy":
            return 1 - np.mean(np.absolute(Y-Y_pred))