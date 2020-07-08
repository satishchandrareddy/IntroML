# driver_casestudy_houseprice.py

import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results
import load_house

# (1) Set up data
nfeature = 3
ntrain_pct = 0.8
Xtrain,Ytrain,Xvalid,Yvalid = load_house.load_house(0.8,True,True,True)
# (2) Define model
np.random.seed(10)
lamb = 0.0001
model = LRegression.LRegression(nfeature,"linear",lamb)
# (3) Compile model
optimizer = Optimizer.GradientDescent(0.5)
model.compile("meansquarederror",optimizer)
# (4) Train model
epochs = 50
history = model.fit(Xtrain,Ytrain,epochs,validation_data=(Xvalid,Yvalid))
# (5) Predictions and plotting
Yvalid_pred = model.predict(Xvalid)
# plot loss and accuracy
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
plt.show()
