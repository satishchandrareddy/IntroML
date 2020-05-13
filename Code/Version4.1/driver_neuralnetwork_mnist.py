# driver_neuralnetwork_mnist.py

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import onehot
import plot_results

# (1) Set up data
file_train = "../Data_MNIST/MNIST_train_set1_30K.csv"
m = 30000
Xtrain,Ytrain = load_mnist.load_mnist(file_train,m)
nclass = 10
Xtrain = Xtrain/255
ntest = 100
Xtest,Ytest = load_mnist.load_mnist("../Data_MNIST/MNIST_test_10K.csv",ntest) 
Xtest = Xtest/255
# (2) Define model
model = NeuralNetwork.NeuralNetwork(784)
model.add_layer(120,"tanh")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = {"method": "Adam", "learning_rate": 0.02, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7}
model.compile("crossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 50
history = model.train(Xtrain,Ytrain,epochs,validation_data=(Xtest,Ytest))
# (5) Predictions and plotting
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_history(history,["accuracy"])
# load test data and predict 
Ytest_pred = model.predict(Xtest)
plot_results.plot_results_mnistdigits_animation(Xtest,Ytest,Ytest_pred,100)
plt.show()