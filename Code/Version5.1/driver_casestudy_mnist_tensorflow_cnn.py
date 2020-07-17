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
# convert data back to 3d images Xtrain_CNN, XValid_CNN have dimensions (#samples,28,28,1)
# Ytrain,Yvalid have dimensions (#samples,1)
XtrainCNN = np.reshape(Xtrain.T,(ntrain,28,28,1))
YtrainCNN = Ytrain.T
XvalidCNN = np.reshape(Xvalid.T,(nvalid,28,28,1))
YvalidCNN = Yvalid.T
# (2) Define model
lamb = 0.0
model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(6,kernel_size=(3,3),input_shape=(28,28,1), activation="relu"),
 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
 tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation="relu"),
 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(128,activation="relu"),
 tf.keras.layers.Dense(84,activation="relu"),
 tf.keras.layers.Dense(nclass,activation="softmax")])
# (3) Compile model
optimizer = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
# (4) Train model
epochs = 10
time_start = time.time()
history = model.fit(XtrainCNN,YtrainCNN,epochs=epochs,batch_size=32,validation_data=(XvalidCNN,YvalidCNN))
time_end = time.time()
print("Train time: {}".format(time_end - time_start))
# (5) Predictions and plotting
# confuation matrix
Afinal = model.predict(XvalidCNN).T
Yvalid_pred = onehot.onehot_inverse(Afinal)
metrics.confusion_matrix(Yvalid,Yvalid_pred,nclass)
# plot loss, accuracy, and animation of results
plot_results.plot_results_history(history.history,["loss","val_loss"])
plot_results.plot_results_history(history.history,["accuracy","val_accuracy"])
plot_results.plot_results_mnist_animation(Xvalid,Yvalid,Yvalid_pred,Afinal,100)
plt.show()
