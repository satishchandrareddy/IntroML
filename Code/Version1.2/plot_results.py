# plot_results.py

import matplotlib.pyplot as plt
import numpy as np

def plot_results_history(history,key_list):
    plt.figure()
    for key in key_list:
        epoch_array = list(range(1,len(history[key])+1))
        plt.plot(epoch_array,history[key],'r-',label=key)
    plt.xlabel("Epoch")
    plt.ylabel(",".join(key_list))
    plt.title(",".join(key_list))
    plt.legend(loc="upper right")

def plot_results_linear(model,Xtrain,Ytrain):
    # plot training data, normal equations solution, machine learning solution
    # determine machine learning prediction
    X0 = Xtrain[0,:]
    X0min = np.min(X0)
    X0max = np.max(X0)
    Xtest = np.array([[X0min,X0max]])
    Ytest_pred = model.predict(Xtest)
    # normal equation solution
    Xb = np.concatenate((Xtrain,np.ones(Ytrain.shape)),axis=0)
    Wb = np.dot(np.dot(Ytrain,Xb.T),np.linalg.inv(np.dot(Xb,Xb.T)))
    W = Wb[0,0]
    b = Wb[0,1]
    Ynorm = W*Xtest+b
    # plot results
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",label="Training Points")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ytest_pred),"r-",label="Machine Learning Prediction")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ynorm),"k-",label="Normal Equation Prediction")
    plt.legend(loc = "upper left")