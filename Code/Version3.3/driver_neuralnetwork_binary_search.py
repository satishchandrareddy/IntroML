# driver_neuralnetwork_binary_search.py

import NeuralNetwork
import example_classification
import numpy as np
import Optimizer
import time
import write_csv

# create function to generate 
def nn(nfeature,lamb,seed,learning_rate,beta):
	np.random.seed(seed)
	model = NeuralNetwork.NeuralNetwork(nfeature)
	model.add_layer(15,"tanh",lamb)
	model.add_layer(11,"tanh",lamb)
	model.add_layer(8,"tanh",lamb)
	model.add_layer(4,"tanh",lamb)
	model.add_layer(1,"sigmoid",lamb)
	optimizer = {"method": "Momentum", "learning_rate": learning_rate, "beta": beta}
	model.compile("binarycrossentropy",optimizer)
	return model

# (1) Set up data
nfeature = 2
m = 1000
case = "quadratic"
nclass = 2
noise = True
np.random.seed(10)
Xtrain,Ytrain,Xvalid,Yvalid = example_classification.example(nfeature,m,case,nclass,noise,0.2)
# loop over hyperparameters
list_save = [["Lambda","Learning Rate", "Batchsize", "Accuracy Train", "Accuracy Valid"]]
list_lamb = [0, 0.01, 0.02]
list_learningrate = [0.01, 0.03, 0.1, 0.3]
list_batchsize = [64, 1000]
seed = 100
beta = 0.9
epochs = 300
for lamb in list_lamb:
	for lr in list_learningrate:
		for batchsize in list_batchsize:
			# create model
			model = nn(nfeature,lamb,seed,lr,beta)
			# train
			history = model.train(Xtrain,Ytrain,epochs,verbose=False,batchsize=batchsize,validation_data=(Xvalid,Yvalid))
			# save results
			list_save.append([lamb, lr, batchsize, history["accuracy"][-1], history["accuracy_valid"][-1]])
			print("Lambda: {} - LR: {} - Batchsize: {} - Acc: {} - Valid Acc: {}".format(lamb,lr,batchsize,history["accuracy"][-1],history["accuracy_valid"][-1]))

outputfile = "Search.csv"
write_csv.write_csv(outputfile,list_save)
