# driver_linearregression.py
#
import matplotlib.pyplot as plt
import numpy as np
import plot_results
import tensorflow as tf

# (1) Set up data
m = 1000
n_features = 2
n_class = 2
X = np.random.randn(n_features,m)
Y = (X[0,:] + X[1,:] - 0.75 > 0).astype(float)
X = np.reshape(X,(m,n_features))
Y = np.reshape(Y,(m,1))
#Y = np.expand_dims(Y,axis=0)
# (2) Define model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid')])
model.compile(loss='binary_crossentropy',optimizer='adam')
model.summary()
epoch = 100
model.fit(X,Y,epoch)
#plot_results.plot_results_logistic(model,X,Y,n_class)
#plt.show()