# driver_neuralnetwork_multiclass.py

import NeuralNetwork
import example_classification
import matplotlib.pyplot as plt
import Optimizer
import plot_results
import time

# (1) Set up data
nfeature = 2
m = 1000
case = "quadratic"
nclass = 3
X,Y = example_classification.example(nfeature,m,case,nclass)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh")
model.add_layer(9,"tanh")
model.add_layer(6,"tanh")
model.add_layer(3,"tanh")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = {"method": "Adam", "learning_rate": 0.02, "beta1": 0.9, "beta2": 0.999,  "epsilon": 1e-8}
model.compile("crossentropy",optimizer)
# (4) Train model
epochs = 100
time_start = time.time()
history = model.fit(X,Y,epochs,batchsize=64)
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Results
# plot loss and accuracy and heatmap in x0-x1 plane
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
plot_results.plot_results_classification((X,Y),model,nclass)
plt.show()