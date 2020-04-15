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
        Y = np.sum(X,axis=0) + 7
        model = LRegression.LRegression(n_feature,"linear")
        optimizer = None
        model.compile(optimizer,"meansquarederror")
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

if __name__ == "__main__":
    unittest.main()
