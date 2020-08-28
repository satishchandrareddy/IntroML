# driver_casestudy_spam_tensorflow.py
# Run in folder IntroML/Code/Version5.1

import load_spam
import matplotlib.pyplot as plt
import metrics
import numpy as np
import plot_results
import text_results
import tensorflow as tf
import time


# (1) Set up data
ntrain_pct = 0.85
Xtrain,Ytrain,Xvalid,Yvalid,Xtrain_raw,Xvalid_raw = load_spam.load_spam(ntrain_pct)
# take transpose of inputs for tensorflow  - sample axis along rows
XtrainT = Xtrain.T
YtrainT = Ytrain.T
XvalidT = Xvalid.T
YvalidT = Yvalid.T
print("XtrainT.shape: {}".format(XtrainT.shape))
print("YtrainT.shape: {}".format(YtrainT.shape))
print("XvalidT.shape: {}".format(XvalidT.shape))
print("YvalidT.shape: {}".format(YvalidT.shape))
# (2) Define model
nfeature = Xtrain.shape[0]
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(200,input_shape=(nfeature,),activation="tanh"),
 tf.keras.layers.Dense(50,activation="tanh"),
 tf.keras.layers.Dense(1,activation="sigmoid")])
# (3) Compile model
optimizer = tf.keras.optimizers.Adam(lr=0.02,beta_1=0.9,beta_2=0.999,epsilon=1e-7)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# (4) Train model
epochs = 20
time_start = time.time()
ntrain = XtrainT.shape[0]
history = model.fit(XtrainT,YtrainT,epochs=epochs,batch_size=ntrain,validation_data=(XvalidT,YvalidT))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
# confusion matrix
Afinal = model.predict(XvalidT).T
Yvalid_pred = np.round(Afinal)
metrics.confusion_matrix(Yvalid,Yvalid_pred,2)
f1score,precision,recall = metrics.f1score(Yvalid,Yvalid_pred)
print("F1Score: {} - Precision: {} - Recall: {}".format(f1score,precision,recall))
text_results.text_results(Yvalid,Yvalid_pred,Xvalid_raw)
# plot loss and accuracy
plot_results.plot_results_history(history.history,["loss","val_loss"])
plot_results.plot_results_history(history.history,["accuracy","val_accuracy"])
plt.show()