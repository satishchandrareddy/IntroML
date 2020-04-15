# plot_results.py

import matplotlib.pyplot as plt
import numpy as np

def plot_results_loss(history):
    plt.figure()
    epoch_array = np.arange(1,history["loss"].shape[0]+1)
    plt.semilogy(epoch_array,history["loss"],'r-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")

def plot_results_linear(model,Xtrain,Ytrain):
    # plot results in plane
    x0 = Xtrain[0,:]
    x0min = np.min(x0)
    x0max = np.max(x0)
    ymin = np.min(Ytrain)
    ymax = np.max(Ytrain)
    Xtest = np.reshape(np.array([x0min,x0max]),(1,2))
    Ytest_pred = model.predict(Xtest)
    # exact solution
    Xb = np.concatenate((Xtrain,np.ones(Ytrain.shape)),axis=0)
    wb = np.dot(np.dot(Ytrain,Xb.T),np.linalg.inv(np.dot(Xb,Xb.T)))
    Xtestb = np.concatenate((Xtest,np.ones((1,2))),axis=0)
    Yb = np.dot(wb,Xtestb)
    # plot regression results
    plt.figure()
    plt.xlim(x0min,x0max)
    plt.ylim(ymin,ymax)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",label="Training Points")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ytest_pred),"r-",label="Model Prediction")
    plt.plot(np.squeeze(Xtest),np.squeeze(Yb),"k-",label="Normal Equation Prediction")
    plt.legend(loc = "upper left")

def plot_results_logistic(model,Xtrain,Ytrain,nclass):
    # plot data and results in 2d
    x0 = Xtrain[0,:]
    x1 = Xtrain[1,:]
    x0min = np.min(x0)
    x0max = np.max(x0)
    x1min = np.min(x1)
    x1max = np.max(x1)
    npoints = 100
    x0lin = np.linspace(x0min,x0max,npoints)
    x1lin = np.linspace(x1min,x1max,npoints)
    x0grid,x1grid = np.meshgrid(x0lin,x1lin)
    x0reshape = np.reshape(x0grid,(1,npoints*npoints))
    x1reshape = np.reshape(x1grid,(1,npoints*npoints))
    yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
    ygrid = np.reshape(yreshape,(npoints,npoints))
    plt.figure()
    plt.pcolormesh(x0grid,x1grid,ygrid)
    plt.colorbar()
    symbol_train = ["ro","bo","go","co"]
    #symbol_test = ["r+","b+","g+","c+"]
    #label = [" train", " test"]
    label = ["train"]
    for count in range(nclass):
        idx_train = np.where(np.squeeze(np.absolute(Ytrain-count))<1e-10)
        strlabeltrain = "Y = " + str(count) + label[0]
        plt.plot(np.squeeze(Xtrain[0,idx_train]),np.squeeze(Xtrain[1,idx_train]),symbol_train[count],label=strlabeltrain)
        #idx_test = np.where(np.squeeze(np.absolute(Ytest-count))<1e-10)
        #strlabeltest = "Y = " + str(count) + label[1]
        #plt.plot(np.squeeze(Xtest[0,idx_test]),np.squeeze(Xtest[1,idx_test]),symbol_test[count],label=strlabeltest)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Heatmap of Training Data and Results")
    plt.legend(loc="upper left")