# unittest_forwardbackprop.py

import example_classification
import NeuralNetwork
import LRegression
import numpy as np
import unittest

class Test_functions(unittest.TestCase):
    
    def test_LinearRegression(self):
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.sum(X,axis=0) + 7
        model = LRegression.LRegression(nfeature,"linear")
        optimizer = None
        model.compile("meansquarederror",optimizer)
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)
  
    def test_LogisticRegression(self):
        nfeature = 2
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
        Y = np.expand_dims(Y,axis=0)
        model = LRegression.LRegression(nfeature,"sigmoid")
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LogisticRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

    def test_NeuralNetwork_binary(self):
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
        Y = np.expand_dims(Y,axis=0)
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(5,"softplus")
        model.add_layer(3,"tanh")
        model.add_layer(1,"sigmoid")
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: 3 layer NeuralNetwork Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

    def test_NeuralNetwork_multi(self):
        nfeature = 2
        m = 1000
        case = "quadratic"
        nclass = 3
        X,Y = example_classification.example(nfeature,m,case,nclass)
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(5,"softplus")
        model.add_layer(3,"tanh")
        model.add_layer(nclass,"softmax")
        optimizer = None
        model.compile("crossentropy",optimizer)
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: 3 layer NeuralNetwork softmax Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

if __name__ == "__main__":
    unittest.main()
