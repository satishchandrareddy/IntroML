# driver_neuralnetwork_multiclass_kfold.py

import NeuralNetwork
import example_classification
import numpy as np
import Optimizer
import time

# create function to generate neural network
def nn(nfeature,seed,learning_rate,beta):
  np.random.seed(seed)
  model = NeuralNetwork.NeuralNetwork(nfeature)
  model.add_layer(11,"tanh")
  model.add_layer(9,"tanh")
  model.add_layer(6,"tanh")
  model.add_layer(3,"tanh")
  model.add_layer(nclass,"softmax")
  optimizer = Optimizer.Momentum(learning_rate,beta)
  model.compile("crossentropy",optimizer)
  return model

# (1) Set up data
nfeature = 2
m = 2000
case = "quadratic"
nclass = 3
noise = False
validperc = 0.1
Xtrain,Ytrain,Xvalid,Yvalid = example_classification.example(nfeature,m,case,nclass,noise,validperc)
seed = 100
lr = 0.03
beta = 0.9
epochs = 300
K = 6
ntrain = Xtrain.shape[1]
fold_size = ntrain // K
train_acc_cv = 0
val_acc_cv = 0
# shuffle data
shuffled_indices = list(range(ntrain))
np.random.shuffle(shuffled_indices)
Xtrain_shuffled, Ytrain_shuffled = Xtrain[:,shuffled_indices], Ytrain[:,shuffled_indices]
print("Number of Folds: {}".format(K))
for k in range(K):
  # split data
  fold_indices = range(k*fold_size,(k+1)*fold_size)
  train_indices = list(set(range(ntrain)) - set(fold_indices))
  Xfold_k, Yfold_k = Xtrain_shuffled[:,fold_indices], Ytrain_shuffled[:,fold_indices]
  Xtrain_k, Ytrain_k = Xtrain_shuffled[:,train_indices], Ytrain_shuffled[:,train_indices]
  # create model
  model = nn(nfeature,seed,lr,beta)
  # train
  ntrain_k = Xtrain_k.shape[1]
  history = model.fit(Xtrain_k,Ytrain_k,epochs,verbose=False,batch_size=ntrain_k,validation_data=(Xfold_k,Yfold_k))
  print("Fold: {}  Accuracy: {}  Valid_Accuracy: {}".format(k,history["accuracy"][-1],history["valid_accuracy"][-1]))
  train_acc_cv += history["accuracy"][-1] / K
  val_acc_cv += history["valid_accuracy"][-1] / K

# compare and print results
print("Acc: {} - Valid Acc: {}".format(train_acc_cv,val_acc_cv))