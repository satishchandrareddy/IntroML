# driver_neuralnetwork_multiclass.py

import NeuralNetwork
import example_classification
import matplotlib.pyplot as plt
import metrics
import Optimizer
import plot_results
import time

# (1) Set up data
nfeature = 2
m = 1000
case = "quadratic"
nclass = 3
noise = True
validperc = 0.2
Xtrain,Ytrain,Xvalid,Yvalid = example_classification.example(nfeature,m,case,nclass,noise,validperc)
# (2) Define model
lamb = 0.02
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh",lamb)
model.add_layer(9,"tanh",lamb)
model.add_layer(6,"tanh",lamb)
model.add_layer(3,"tanh",lamb)
model.add_layer(nclass,"softmax",lamb)
# (3) Compile model
#optimizer = Optimizer.GradientDescent(0.3)
#optimizer = Optimizer.Momentum(0.3,0.9)
#optimizer = Optimizer.RmsProp(0.02,0.9,1e-8)
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-8)
model.compile("crossentropy",optimizer)
# (4) Train model
epochs = 100
time_start = time.time()
history = model.fit(Xtrain,Ytrain,epochs,batch_size=64,validation_data=(Xvalid,Yvalid))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Results
# confusion matrix
Yvalid_pred = model.predict(Xvalid)
metrics.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification((Xtrain,Ytrain),model,nclass,validation_data=(Xvalid,Yvalid))
plt.show()