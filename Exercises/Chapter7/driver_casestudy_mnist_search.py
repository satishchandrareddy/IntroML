# driver_neuralnetwork_mnist_search.py
# Run in folder IntroML/Code/Version4.1

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import metrics
import Optimizer
import plot_results
import time
import write_csv

# create function to generate neural network
def nn(nfeature,nclass,nunits,lamb,seed,learning_rate,beta1,beta2,epsilon):
  np.random.seed(seed)
  model = NeuralNetwork.NeuralNetwork(nfeature)
  model.add_layer(nunits,"relu",lamb)
  model.add_layer(nclass,"softmax",lamb)
  optimizer = Optimizer.Adam(learning_rate,beta1,beta2,epsilon)
  model.compile("crossentropy",optimizer)
  return model

# Set up data
ntrain = 6000
nvalid = 1000
nclass = 10
Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(ntrain,nvalid)
nfeature = Xtrain.shape[0]
# loop over hyperparameters
list_save = [["Hidden Units","Lambda","Learning Rate", "batch_size", "Train Accuracy", "Valid Accuracy"]]
list_nunits = [64, 128]
list_lamb = [0, 0.01]
list_learningrate = [0.001, 0.005, 0.02]
list_batch_size = [128, ntrain]
seed = 10
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
epochs = 40
for nunits in list_nunits:
	for lamb in list_lamb:
		for lr in list_learningrate:
			for batch_size in list_batch_size:
				# create model
				model = nn(nfeature,nclass,nunits,lamb,seed,lr,beta1,beta2,epsilon)
				# train
				history = model.fit(Xtrain,Ytrain,epochs,verbose=False,batch_size=batch_size,validation_data=(Xvalid,Yvalid))
				# save results
				list_save.append([nunits, lamb, lr, batch_size, history["accuracy"][-1], history["valid_accuracy"][-1]])
				print("Hidden Units: {} - Lambda: {} - LR: {} - batch_size: {} - Acc: {} - Valid Acc: {}".format(nunits,lamb,lr,batch_size,history["accuracy"][-1],history["valid_accuracy"][-1]))
outputfile = "Search_mnist.csv"
write_csv.write_csv(outputfile,list_save)