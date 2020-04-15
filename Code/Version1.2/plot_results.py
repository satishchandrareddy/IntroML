# plot_results.py

import matplotlib.pyplot as plt
import numpy as np

def plot_results_history(history,key):
    plt.figure()
    epoch_array = np.arange(0,history[key].shape[0])
    plt.plot(epoch_array,history["loss"],'r-')
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(key)

def plot_results_linear(model,Xtrain,Ytrain):
    # plot results in plane
    X0 = Xtrain[0,:]
    X0min = np.min(X0)
    X0max = np.max(X0)
    Xtest = np.reshape(np.array([X0min,X0max]),(1,2))
    Ytest_pred = model.predict(Xtest)
    # exact solution
    Xb = np.concatenate((Xtrain,np.ones(Ytrain.shape)),axis=0)
    wb = np.dot(np.dot(Ytrain,Xb.T),np.linalg.inv(np.dot(Xb,Xb.T)))
    Xtestb = np.concatenate((Xtest,np.ones((1,2))),axis=0)
    Yb = np.dot(wb,Xtestb)
    # plot regression results
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",label="Training Points")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ytest_pred),"r-",label="Model Prediction")
    plt.plot(np.squeeze(Xtest),np.squeeze(Yb),"k-",label="Normal Equation Prediction")
    plt.legend(loc = "upper left")