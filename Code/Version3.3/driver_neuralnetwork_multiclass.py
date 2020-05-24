# driver_neuralnetwork_multiclass.py

import NeuralNetwork
import example_classification
import matplotlib.pyplot as plt
import numpy as np
import onehot
import Optimizer
import plot_results

# (1) Set up data
nfeature = 2
m = 1000
case = "quadratic"
nclass = 3
noise = False
Xtrain,Ytrain,Xvalid,Yvalid = example_classification.example(nfeature,m,case,nclass,noise,0.1)
# (2) Define model
lamb = 0.0
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh",lamb)
model.add_layer(9,"tanh",lamb)
model.add_layer(6,"tanh",lamb)
model.add_layer(3,"tanh",lamb)
model.add_layer(nclass,"softmax",lamb)
# (3) Compile model
optimizer = {"method": "Momentum", "learning_rate": 0.05, "beta": 0.9}
model.compile("crossentropy",optimizer)
# (4) Train model
epochs = 100
history = model.train(Xtrain,Ytrain,epochs,batchsize=32,validation_data=(Xvalid,Yvalid))
# (5) Results
# confusion matrix
Yvalid_pred = model.predict(Xvalid)
model.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","loss_valid"])
plot_results.plot_results_history(history,["accuracy","accuracy_valid"])
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification((Xtrain,Ytrain),model,nclass,validation_data=(Xvalid,Yvalid))
plt.show()