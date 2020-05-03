# driver_neuralnetwork.py

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import onehot
import plot_results

# (1) Set up data
file_train = "Data/MNIST_train_set1_30K.csv"
m = 30000
X,Y = load_mnist.load_mnist(file_train,m)
nclass = 10
Y_onehot = onehot.onehot(Y,nclass)
X = X/255
# (2) Define model
model = NeuralNetwork.NeuralNetwork(784)
model.add_layer(120,"tanh")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = {"method": "Adam", "learning_rate": 0.02, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7}
model.compile("crossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 100
history = model.train(X,Y_onehot,epochs)
# (5) Plot results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
plt.show()