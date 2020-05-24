# driver_linearregression.py

import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
m = 1000
X = np.random.rand(1,m)
Y = 0.5*X + 0.25
Y = Y + 0.1*np.random.randn(m)
# (2) Define model
lamb = 0.01
model = LRegression.LRegression(1,"linear",lamb)
# (3) Compile model
optimizer = {"method": "GradientDescent", "learning_rate": 0.5}
model.compile("meansquarederror",optimizer)
# (4) Train model
epochs = 50
history = model.train(X,Y,epochs)
# (5) Results
# plot results
plot_results.plot_results_linear(model,X,Y)
# plot loss
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
plt.show()