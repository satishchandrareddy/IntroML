# driver_casestudy_houseprice.py

import LRegression
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results
import load_house

# (1) Set up data
m = 3
ntrain_pct = 0.8
Xtrain,Ytrain,Xvalid,Yvalid = load_house.load_house(0.8,
													transform=True, 
													standardize=True)
# (2) Define model
lamb = 0.0001
model = LRegression.LRegression(m,"linear",lamb)
# (3) Compile model
optimizer = Optimizer.GradientDescent(0.5)
model.compile("meansquarederror",optimizer)
# (4) Train model
epochs = 50
history = model.fit(Xtrain,Ytrain,epochs,validation_data=(Xvalid,Yvalid))
# (5) Predictions and plotting
Yvalid_pred = model.predict(Xvalid)
# plot loss
plot_results.plot_results_history(history,["loss","valid_loss"])
plot_results.plot_results_history(history,["accuracy","valid_accuracy"])
plt.show()
