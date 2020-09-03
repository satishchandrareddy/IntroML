#unittest_forwardbackprop.py

import LRegression
import numpy as np
import unittest

class Test_functions(unittest.TestCase):
    
    def test_LinearRegression(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.random.rand(1,m)
        # (2) define object
        model = LRegression.LRegression(nfeature,"linear")
        # (3) compile
        optimizer = None
        model.compile("meansquarederror",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        # (5) assert statement
        self.assertLessEqual(error,1e-7)

    def test_LinearRegression_logcosh(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.random.rand(1,m)
        # (2) define object
        model = LRegression.LRegression(nfeature,"linear")
        # (3) compile
        optimizer = None
        model.compile("logcosh",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        # (5) assert statement
        self.assertLessEqual(error,1e-7)
  
    def test_LogisticRegression(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 2
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.round(np.random.rand(1,m))
        # (2) define object
        model = LRegression.LRegression(nfeature,"sigmoid")
        # (3) compile
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LogisticRegression Error: {}".format(error))
        # assert statement
        self.assertLessEqual(error,1e-7)

    def test_LogisticRegression_mirrorsigmoid(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 2
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.round(np.random.rand(1,m))
        # (2) define object
        model = LRegression.LRegression(nfeature,"mirrorsigmoid")
        # (3) compile
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LogisticRegression Error: {}".format(error))
        # assert statement
        self.assertLessEqual(error,1e-7)

if __name__ == "__main__":
    unittest.main()