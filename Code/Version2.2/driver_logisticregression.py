# driver_logisticregression.py

import LRegression
import example_classification
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
nfeature = 2
m = 1000
case = "linear"
nclass = 2
X,Y = example_classification.example(nfeature,m,case,nclass)
# (2) Define model
model = LRegression.LRegression(nfeature,"sigmoid")
# (3) Compile model
optimizer = Optimizer.GradientDescent(0.5)
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