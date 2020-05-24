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

    def train(self,X,Y,epochs,**kwargs):
        # iterate over epochs
        loss = []
        accuracy = []
        if "validation_data" in kwargs:
            loss_valid = []
            accuracy_valid = []
        #get mini-batches
        if "batchsize" in kwargs:
            mini_batch = self.mini_batch(X,Y,kwargs["batchsize"])
        else:
            mini_batch = [(X,Y)]
        # train
        for epoch in range(epochs):
            # train using mini-batches
            for (Xbatch,Ybatch) in mini_batch:
                self.forward_propagate(Xbatch)
                self.back_propagate(Xbatch,Ybatch)
                self.update_param()
            # compute loss and accuracy after cycling through mini-batches
            Y_pred = self.predict(X)
            loss.append(self.compute_loss(Y))
            accuracy.append(self.accuracy(Y,Y_pred))
            # compute loss and accuracy for test set
            if "validation_data" in kwargs:
                self.forward_propagate(kwargs["validation_data"][0])
                loss_valid.append(self.compute_loss(kwargs["validation_data"][1]))
                Ytest_pred = self.predict(kwargs["validation_data"][0])
                accuracy_valid.append(self.accuracy(kwargs["validation_data"][1],Ytest_pred))
                print("Epoch: {} - loss: {} - accuracy: {} - loss_valid: {} - accuracy_valid: {}".format(epoch+1,loss[epoch],accuracy[epoch],loss_valid[epoch],accuracy_valid[epoch]))
            else:
                print("Epoch: {} - Cost: {} - Accuracy: {}".format(epoch,loss[epoch],accuracy[epoch]))
        if "validation_data" in kwargs:
            return {"loss":np.array(loss),"accuracy":np.array(accuracy),"loss_valid":np.array(loss_valid),"accuracy_valid":np.array(accuracy_valid)}
        else:
            return {"loss":np.array(loss),"accuracy":np.array(accuracy)}


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

    def f1score(self,Y,Y_pred):
        idx_truepositive = np.where((np.absolute(Y-1)<1e-7)&(np.absolute(Y_pred-1)<1e-7))
        idx_actualpositive = np.where(np.absolute(Y-1)<1e-7)
        idx_predpositive = np.where(np.absolute(Y_pred-1)<1e-7)
        precision = np.size(idx_truepositive)/(np.size(idx_predpositive)+1e-7)
        recall = np.size(idx_truepositive)/(np.size(idx_actualpositive)+1e-7)
        f1score = 2*precision*recall/(precision + recall)
        return f1score,precision,recall
           
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

    def mini_batch(self,X,Y,batchsize):
        m = Y.shape[1]
        # determine number of mini-batches
        if m % batchsize == 0:
            n = int(m/batchsize)
        else:
            n = int(m/batchsize) + 1
        # create mini-batches
        mini_batch = []
        for count in range(n):
            start = count*batchsize
            end = start + min(start+batchsize,m)
            mini_batch.append((X[:,start:end],Y[:,start:end]))
        return mini_batch

    def confusion_matrix(self,Y,Y_pred,nclass):
        print("\t\tConfusion Matrix")
        print("\t\tActual")
        strhead = [" \t\t"] + [str(i)+"\t" for i in range(nclass)]
        print("".join(strhead))
        for pred in range(nclass):
            idx_pred = np.where(np.squeeze(np.absolute(Y_pred-pred)<1e-7))
            if pred == 0:
                str_row = ["Predicted  0\t"]
            else:
                str_row = ["           "+str(pred)+"\t"]
            for actual in range(nclass):
                idx_act = np.where(np.squeeze(np.absolute(Y[0,idx_pred]-actual)<1e-7))
                str_row.append(str(np.size(idx_act))+"\t")
            print("".join(str_row))