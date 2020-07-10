# driver_neuralnetwork_mnist.py

import load_mnist
import matplotlib.pyplot as plt
import metrics
import numpy as np
import onehot
import plot_results
import tensorflow as tf
import time

# (1) Set up data
ntrain = 6000
nvalid = 1000
nclass = 10
Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(ntrain,nvalid)
# take transpose of inputs for tensorflow  - sample axis along rows
XtrainT = Xtrain.T
YtrainT = Ytrain.T
XvalidT = Xvalid.T
YvalidT = Yvalid.T
# (2) Define model
lamb = 0.0
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(128,input_shape=(784,), activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(lamb)),
 tf.keras.layers.Dense(nclass,activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(lamb))])
# (3) Compile model
optimizer = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
# (4) Train model
epochs = 40
time_start = time.time()
history = model.fit(XtrainT,YtrainT,epochs=epochs,batch_size=ntrain,validation_data=(XvalidT,YvalidT))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
# confusion matrix
Afinal = model.predict(XvalidT).T
Yvalid_pred = onehot.onehot_inverse(Afinal)
metrics.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss, accuracy, and animation of results
plot_results.plot_results_history(history.history,["loss","val_loss"])
plot_results.plot_results_history(history.history,["accuracy","val_accuracy"])
plot_results.plot_results_mnist_animation(Xvalid,Yvalid,Yvalid_pred,Afinal,100)
plt.show()
