# plot_results.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_results_history(history,key_list):
    plt.figure()
    linemarker = ["r-","b-","k-","g-","c-"]
    for count in range(len(key_list)):
        epoch_array = list(range(1,len(history[key_list[count]])+1))
        plt.plot(epoch_array,history[key_list[count]],linemarker[count],label=key_list[count])
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

def plot_results_classification(data_train,model,nclass=2,**kwargs):
    if "validation_data" in kwargs:
        plot_results_data(data_train,nclass,validation_data=kwargs["validation_data"])
    else:
        plot_results_data(data_train,nclass)
    plot_results_heatmap(model,data_train[0])

def plot_results_data(data_train,nclass=2,**kwargs):
    # plot training data - loop over labels (0, 1) and points in dataset which have those labels
    Xtrain = data_train[0]
    Ytrain = data_train[1]
    plt.figure()
    symbol_train = ["ro","bo","go","co","yo"]
    for count in range(nclass):
        idx_train = np.where(np.squeeze(np.absolute(Ytrain-count))<1e-10)
        strlabeltrain = "Y = " + str(count) + " train"
        plt.plot(np.squeeze(Xtrain[0,idx_train]),np.squeeze(Xtrain[1,idx_train]),symbol_train[count],label=strlabeltrain)
    if "validation_data" in kwargs:
        Xvalid = kwargs["validation_data"][0]
        Yvalid = kwargs["validation_data"][1]
        symbol_valid = ["r^","b^","g^","c^","y^"]
        for count in range(nclass):
            idx_valid = np.where(np.squeeze(np.absolute(Yvalid-count))<1e-10)
            strlabelvalid = "Y = " + str(count) + " valid"
            plt.plot(np.squeeze(Xvalid[0,idx_valid]),np.squeeze(Xvalid[1,idx_valid]),symbol_valid[count],label=strlabelvalid)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(loc="upper left")
    plt.title("Data")

def plot_results_heatmap(model,Xtrain):
    # plot heat map of model results
    x0 = Xtrain[0,:]
    x1 = Xtrain[1,:]
    npoints = 100
    # create 1d grids in x0 and x1 directions
    x0lin = np.linspace(np.min(x0),np.max(x0),npoints)
    x1lin = np.linspace(np.min(x1),np.max(x1),npoints)
    # create 2d grads for x0 and x1 and reshape into 1d grids 
    x0grid,x1grid = np.meshgrid(x0lin,x1lin)
    x0reshape = np.reshape(x0grid,(1,npoints*npoints))
    x1reshape = np.reshape(x1grid,(1,npoints*npoints))
    # predict results (concatenated x0 and x1 1-d grids to create feature matrix)
    yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
    # reshape results into 2d grid and plot heatmap
    heatmap = np.reshape(yreshape,(npoints,npoints))
    plt.pcolormesh(x0grid,x1grid,heatmap)
    plt.colorbar()
    plt.title("Data and Heatmap of Prediction Results")

def plot_results_mnist(X,Y,Y_pred,Afinal,idx):
    plt.figure()
    plt.subplot(121)
    npixel_width = 28
    npixel_height = 28
    # create 1-d grids for x and and y directions
    npixel_width = 28
    npixel_height = 28
    x0 = np.linspace(0,1,npixel_width)
    x1 = np.linspace(0,1,npixel_height)
    x0grid,x1grid = np.meshgrid(x0,x1)
    digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
    plt.pcolormesh(x0grid,x1grid,digit_image,cmap="Greys")
    plt.text(0.5,1.01,"Image: {0} Actual: {1} Predicted: {2}".format(idx,Y[0,idx],Y_pred[0,idx]),
             size=10,ha="center")
    plt.subplot(122)
    label = [str(idx) for idx in range(Afinal.shape[0])]
    plt.bar(np.arange(Afinal.shape[0]),Afinal[:,idx],tick_label=label)
    plt.ylim(0,1)
    plt.text(4.5,1.01,"Probability of Prediction",size=10,ha="center")

def plot_results_mnist_animation(X,Y,Y_pred,nframe):
    # number of data points
    m = X.shape[1]
    # determine number of frames
    nplot = min(m,nframe)
    # set up plot
    fig,ax = plt.subplots()
    # create 1-d grids for x and and y directions
    npixel_width = 28
    npixel_height = 28
    x0 = np.linspace(0,1,npixel_width)
    x1 = np.linspace(0,1,npixel_height)
    x0grid,x1grid = np.meshgrid(x0,x1)
    # details for barchart
    container = []
    for idx in range(nplot):
        # need flipud or else image will be upside down
        digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
        pc = ax.pcolormesh(x0grid,x1grid,digit_image,cmap="Greys")
        title = ax.text(0.5,1.05,"Image: {0} Actual: {1}  Predicted: {2}".format(idx,Y[0,idx],Y_pred[0,idx]),
                            size=plt.rcParams["axes.titlesize"],ha="center")
        #xticks = ax2.xticks(position,label)
        container.append([pc,title])
    ani = animation.ArtistAnimation(fig,container,interval = 1000, repeat = False, blit=False)
    plt.show()
    plt.close()



if __name__ == "__main__":
    Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(100,100)
    plot_results_mnist_animation(Xvalid,Yvalid,Yvalid,100)