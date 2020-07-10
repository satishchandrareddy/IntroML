# driver_neuralnetwork_mnist.py

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import metrics
import Optimizer
import plot_results
import time

# (1) Set up data
ntrain = 6000
nvalid = 1000
nclass = 10
Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(ntrain,nvalid)
# (2) Define model
np.random.seed(10)
lamb = 0.0
model = NeuralNetwork.NeuralNetwork(784)
model.add_layer(128,"tanh",lamb)
model.add_layer(nclass,"softmax",lamb)
# (3) Compile model
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-7)
model.compile("crossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 40
time_start = time.time()
history = model.fit(Xtrain,Ytrain,epochs,batch_size=ntrain,validation_data=(Xvalid,Yvalid))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
# confusion matrix
Yvalid_pred = model.predict(Xvalid)
metrics.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss, accuracy, and animation of results
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
plot_results.plot_results_mnist_animation(Xvalid,Yvalid,Yvalid_pred,model.get_Afinal(),100)
plt.show()