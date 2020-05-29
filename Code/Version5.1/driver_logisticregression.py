# driver_logisticregression.py

import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
m = 1000
nfeature = 2
nclass=2
X = np.random.randn(nfeature,m)
Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
Y = np.expand_dims(Y,axis=0)
# (2) Define model
lamb = 0.01
model = LRegression.LRegression(nfeature,"sigmoid",lamb)
# (3) Compile model
optimizer = {"method": "GradientDescent", "learning_rate": 0.5}
model.compile("binarycrossentropy",optimizer)
# (4) Train model
epochs = 100
history = model.fit(X,Y,epochs)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification((X,Y),model,nclass)
plt.show()