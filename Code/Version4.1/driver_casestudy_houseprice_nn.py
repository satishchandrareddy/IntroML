# driver_casestudy_houseprice_nn.py

import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results
import load_house
import time


# (1) Set up data
nfeature = 3
ntrain_pct = 0.8
Xtrain,Ytrain,Xvalid,Yvalid = load_house.load_house(0.8,False,True,True)
# (2) Define model
np.random.seed(10)
lamb = 0.001
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(16,"relu",lamb)
model.add_layer(1,"linear",lamb)
# (3) Compile model
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-7)
model.compile("meansquarederror",optimizer)
model.summary()
# (4) Train model
epochs = 40
ntrain = Xtrain.shape[0]
time_start = time.time()
history = model.fit(Xtrain,Ytrain,epochs,batch_size=ntrain,validation_data=(Xvalid,Yvalid))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
Yvalid_pred = model.predict(Xvalid)
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
plt.show()