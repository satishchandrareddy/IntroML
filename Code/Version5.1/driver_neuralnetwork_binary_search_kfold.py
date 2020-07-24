# driver_neuralnetwork_binary_search_kfold.py

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
  optimizer = Optimizer.Momentum(learning_rate,beta)
  model.compile("binarycrossentropy",optimizer)
  return model

# (1) Set up data
nfeature = 2
m = 1200
case = "quadratic"
nclass = 2
noise = True
np.random.seed(10)
Xtrain,Ytrain = example_classification.example(nfeature,m,case,nclass,noise)
# loop over hyperparameters
list_save = [["Lambda","Learning Rate", "Train Accuracy", "Valid Accuracy"]]
list_lamb = [0, 0.01, 0.02]
list_learningrate = [0.01, 0.03, 0.1, 0.3]
list_batch_size = [64, 1000]
seed = 100
beta = 0.9
epochs = 100
K = 6
ntrain = Xtrain.shape[1]
fold_size = ntrain // K
for lamb in list_lamb:
  for lr in list_learningrate:
    for batch_size in list_batch_size:
      train_acc = 0
      val_acc = 0
      # shuffle data
      shuffled_indices = list(range(ntrain))
      np.random.shuffle(shuffled_indices)
      Xtrain_shuffled, Ytrain_shuffled = Xtrain[:,shuffled_indices], Ytrain[:,shuffled_indices]
      for k in range(K):
        # split data
        fold_indices = range(k*fold_size,(k+1)*fold_size)
        train_indices = list(set(range(ntrain)) - set(fold_indices))
        Xfold_k, Yfold_k = Xtrain_shuffled[:,fold_indices], Ytrain_shuffled[:,fold_indices]
        Xtrain_k, Ytrain_k = Xtrain_shuffled[:,train_indices], Ytrain_shuffled[:,train_indices]
        # create model
        model = nn(nfeature,lamb,seed,lr,beta)
        # train
        history = model.fit(Xtrain_k,Ytrain_k,epochs,verbose=False,batch_size=batch_size,validation_data=(Xfold_k,Yfold_k))
        train_acc += history["accuracy"][-1] / K
        val_acc += history["valid_accuracy"][-1] / K

      # save results
      list_save.append([lamb, lr, batch_size, train_acc, val_acc])
      print("Lambda: {} - LR: {}- batch_size: {} - Acc: {} - Valid Acc: {}".format(lamb,lr,batch_size,train_acc,val_acc))

outputfile = "Search_KFold.csv"
write_csv.write_csv(outputfile,list_save)