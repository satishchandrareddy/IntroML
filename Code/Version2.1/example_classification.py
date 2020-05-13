# example_classification.py

import matplotlib.pyplot as plt
import numpy as np

def example(nfeature,m,case,nclass=2):
	X = 4*np.random.rand(nfeature,m)-2
	if case == "linear":
		Y = X[0,:] + X[1,:] - 0.25
	elif case == "quadratic":
		Y = X[1,:] - np.square(X[0,:]) + 1.5
	elif case == "cubic":
		Y = X[1,:] - np.power(X[0,:],3) - 2*np.power(X[0,:],2)+ 1.5
	elif case == "disk":
		Y = np.square(X[0,:])+np.square(X[1,:])-1
	elif case == "ring":
		Y = 1.25*np.sqrt(np.square(X[0,:])+np.square(X[1,:]))
		Y = np.fmod(Y,nclass)
	elif case == "band":
		Y = X[0,:] + X[1,:] 
		Y = np.fmod(Y,nclass)
	Y = np.maximum(Y,0.0)
	Y = np.round(Y)
	Y = np.minimum(Y,nclass-1)
	Y = np.expand_dims(Y,axis=0)
	return X,Y

def plot_results_classification(Xtrain,Ytrain,nclass):
    # plot heat map of model results
    plt.figure()
    # plot training data - loop over labels plot points in dataset
    # Y=0 points (red) and Y=1 points (blue)
    symbol_train = ["ro","bo","co","go"]
    for count in range(nclass):
        idx_train = np.where(np.squeeze(np.absolute(Ytrain-count))<1e-10)
        strlabeltrain = "Y = " + str(count) + " train"
        plt.plot(np.squeeze(Xtrain[0,idx_train]),np.squeeze(Xtrain[1,idx_train]),symbol_train[count],label=strlabeltrain)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Training Data")

if __name__ == "__main__":
	nfeature = 2
	m = 5000
	case = "band"
	nclass = 2
	X,Y = example(nfeature,m,case,nclass)
	plot_results_classification(X,Y,nclass)
	plt.show()