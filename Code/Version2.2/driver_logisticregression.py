# driver_logisticregression.py

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
# (3) Compile model
optimizer = {"method": "GradientDescent", "learning_rate": 0.5}
model.compile("binarycrossentropy",optimizer)
# (4) Train model
epochs = 100
history = model.train(X,Y,epochs)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification(model,X,Y,n_class)
plt.show()