# driver_casestudy_spam_logistic.py

import load_spam
import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import metrics
import Optimizer
import plot_results
import text_results
import time

# (1) Set up data
ntrain_pct = 0.85
Xtrain,Ytrain,Xvalid,Yvalid,Xtrain_raw,Xvalid_raw = load_spam.load_spam(ntrain_pct)
# (2) Define model
nfeature = Xtrain.shape[0]
np.random.seed(10)
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(1,"sigmoid",lamb=0.0005)
# (3) Compile model
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-7)
model.compile("binarycrossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 50
time_start = time.time()
ntrain = Xtrain.shape[1]
history = model.fit(Xtrain,Ytrain,epochs,batch_size=ntrain,validation_data=(Xvalid,Yvalid))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
# confusiont matrix
print("Metrics for Validation Dataset")
Yvalid_pred = model.predict(Xvalid)
metrics.confusion_matrix(Yvalid,Yvalid_pred,2)
f1score,precision,recall = metrics.f1score(Yvalid,Yvalid_pred)
print("F1Score: {} - Precision: {} - Recall: {}".format(f1score,precision,recall))
text_results.text_results(Yvalid,Yvalid_pred,Xvalid_raw)
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
plt.show()