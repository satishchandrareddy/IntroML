#unittest_forwardbackprop.py

import LRegression
import numpy as np
import unittest

class Test_functions(unittest.TestCase):
    
    def test_LinearRegression(self):
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.sum(X,axis=0,keepdims=True) + 7
        model = LRegression.LRegression(nfeature,"linear")
        optimizer = None
        model.compile("meansquarederror",optimizer)
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

if __name__ == "__main__":
    unittest.main()
