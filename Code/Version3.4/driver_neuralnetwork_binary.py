# driver_neuralnetwork_binary.py

import NeuralNetwork
import example_classification
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
nfeature = 2
m = 1000
case = "quadratic"
nclass = 2
Xtrain,Ytrain,Xvalid,Yvalid = example_classification.example(nfeature,m,case,nclass,0.1)
# (2) Define model
lamb = 0.02
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh",lamb)
model.add_layer(8,"tanh",lamb)
model.add_layer(4,"tanh",lamb)
model.add_layer(1,"sigmoid",lamb)
# (3) Compile model and print summary
optimizer = {"method": "GradientDescent", "learning_rate": 0.1}
model.compile("binarycrossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 100
history = model.train(Xtrain,Ytrain,epochs,batchsize=32,validation_data=(Xvalid,Yvalid))
Yvalid_pred = model.predict(Xvalid)
f1score,precision,recall=model.f1score(Yvalid,Yvalid_pred)
print()
# (5) Results
# f1score
Yvalid_pred = model.predict(Xvalid)
f1score,precision,recall=model.f1score(Yvalid,Yvalid_pred)
print("F1Score: {} - Precision: {} - Recall: {}".format(f1score,precision,recall))
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","loss_valid"])
plot_results.plot_results_history(history,["accuracy","accuracy_valid"])
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification(model,Xtrain,Ytrain)
plt.show()