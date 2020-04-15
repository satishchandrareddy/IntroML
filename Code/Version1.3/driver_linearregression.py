# driver_linearregression.py
#
import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
m = 1000
X = np.random.rand(1,m)
Y = 0.5*X + 0.25
Y = Y + 0.1*np.random.randn(1,m)
# (2) Define model
model = LRegression.LRegression(1,"linear")
optimizer = Optimizer.GradientDescent(0.1)
model.compile(optimizer,"meansquarederror")
# (3) Fit model
epochs = 200
history = model.train(X,Y,epochs)
# (4) Results
# plot results
plot_results.plot_results_linear(model,X,Y)
# plot loss
plot_results.plot_results_loss(history)
plt.show()