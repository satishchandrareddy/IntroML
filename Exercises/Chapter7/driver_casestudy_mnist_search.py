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
def nn(nfeature,nunits,activation_func,lamb,seed,learning_rate,beta1,beta2,epsilon):
  np.random.seed(seed)
  model = NeuralNetwork.NeuralNetwork(nfeature)
  model.add_layer(nunits,activation_func,lamb)
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
list_save = [["Hidden Units","Activation Function","Lambda","Learning Rate", "batch_size", "Train Accuracy", "Valid Accuracy"]]
list_nunits = [32, 64, 128]
list_act_funcs = ["relu", "tanh"]
list_lamb = [0, 0.01, 0.1, 1, 10]
list_learningrate = [0.01, 0.1]
list_batch_size = [1000, ntrain]
seed = 100
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
epochs = 40
best_valid_acc = 0
best_model = None
history_best = None
for nunits in list_nunits:
	for act_func in list_act_funcs:
		for lamb in list_lamb:
			for lr in list_learningrate:
				for batch_size in list_batch_size:
					# create model
					model = nn(nfeature,nunits,act_func,lamb,seed,lr,beta1,beta2,epsilon)
					# train
					history = model.fit(Xtrain,Ytrain,epochs,verbose=False,batch_size=batch_size,validation_data=(Xvalid,Yvalid))
					# save results
					train_acc = history["accuracy"][-1]
					valid_acc = history["valid_accuracy"][-1]
					if valid_acc > best_valid_acc:
						best_valid_acc = valid_acc 
						best_model = model
						history_best = history

					list_save.append([nunits, act_func, lamb, lr, batch_size, train_acc, valid_acc])
					print("Hidden Units: {} - Activation Function: {} - Lambda: {} - LR: {} - batch_size: {} - Acc: {} - Valid Acc: {}".format(nunits,act_func,lamb,lr,batch_size,history["accuracy"][-1],history["valid_accuracy"][-1]))

outputfile = "Search_mnist.csv"
write_csv.write_csv(outputfile,list_save)

best_model.summary()

# Predictions and plotting
# confusion matrix
print("Metrics for Validation Dataset")
Yvalid_pred = best_model.predict(Xvalid)
metrics.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss, accuracy, and animation of results
plot_results.plot_results_history(history_best,["loss","valid_loss"])
plot_results.plot_results_history(history_best,["accuracy","valid_accuracy"])
plot_results.plot_results_mnist_animation(Xvalid,Yvalid,Yvalid_pred,best_model.get_Afinal(),100)
plt.show()