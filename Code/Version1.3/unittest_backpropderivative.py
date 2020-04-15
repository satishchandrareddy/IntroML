'''
unittest_backpropderivative.py
'''
import LRegression
import numpy as np
import unittest

class Test_functions(unittest.TestCase):
    
    def test_LinearRegression(self):
        m = 1000
        n_feature = 8
        X = np.random.rand(n_feature,m)
        Y = np.sum(X,axis=0,keepdims=True) + 7
        model = LRegression.LRegression(n_feature,"linear")
        optimizer = None
        model.compile(optimizer,"meansquarederror")
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("backpropderivative: LinearRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)
  
    def test_LogisticRegression(self):
        m = 1000
        n_feature = 2
        X = np.random.rand(n_feature,m)
        Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
        #Y = np.expand_dims(Y,axis=0)
        model = LRegression.LRegression(n_feature,"sigmoid")
        optimizer = None
        model.compile(optimizer,"binarycrossentropy")
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("backpropderivative: LogisticRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)


if __name__ == "__main__":
    unittest.main()
