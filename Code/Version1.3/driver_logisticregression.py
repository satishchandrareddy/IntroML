# driver_linearregression.py
#
import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
m = 1000
n_features = 2
n_class = 2
X = np.random.randn(n_features,m)
Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
Y = np.expand_dims(Y,axis=0)
# (2) Define model
model = LRegression.LRegression(2,"sigmoid")
optimizer = Optimizer.GradientDescent(2.0)
model.compile(optimizer,"binarycrossentropy")
epochs = 300
print("Before fit")
model.train(X,Y,epochs)
print("After fit")
W = model.get_params(0,"params","W")
b = model.get_params(0,"params","b")
print("W: {}".format(W))
print("b: {}".format(b))
plot_results.plot_results_logistic(model,X,Y,n_class)
plt.show()