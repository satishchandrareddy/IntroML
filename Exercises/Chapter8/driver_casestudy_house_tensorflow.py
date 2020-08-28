# driver_casestudy_house_tensorflow.py
# Run in folder IntroML/Code/Version5.1

import matplotlib.pyplot as plt
import numpy as np
import plot_results
import load_house
import tensorflow as tf

# (1) Set up data
ntrain_pct = 0.8
Xtrain,Ytrain,Xvalid,Yvalid = load_house.load_house(0.8,True,True,True)
# take transpose of inputs for tensorflow  - sample axis along rows
XtrainT = Xtrain.T
YtrainT = Ytrain.T
XvalidT = Xvalid.T
YvalidT = Yvalid.T
print("XtrainT.shape: {} - YtrainT.shape: {}".format(XtrainT.shape,YtrainT.shape))
print("XvalidT.shape: {} - YvalidT.shape: {}".format(XvalidT.shape,YvalidT.shape))
# (2) Define model
nfeature = Xtrain.shape[0]
lamb = 0.0001
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1,input_shape=(nfeature,),
                        kernel_regularizer=tf.keras.regularizers.l2(lamb))
])
# (3) Compile model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])
# (4) Train model
epochs = 50
ntrain = XtrainT.shape[0]
history = model.fit(XtrainT,YtrainT,epochs=epochs,batch_size=ntrain,validation_data=(XvalidT,YvalidT))
# (5) Predictions and plotting
Yvalid_pred = model.predict(XvalidT)
# plot loss and accuracy (mean absolute error for linear regression)
plot_results.plot_results_history(history.history,["loss","val_loss"])
plot_results.plot_results_history(history.history,["mean_absolute_error","val_mean_absolute_error"])
plt.show()