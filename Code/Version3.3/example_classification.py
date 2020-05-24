# example_classification.py

import matplotlib.pyplot as plt
import numpy as np
import plot_results

def example(nfeature,m,case,nclass=2,noise=False,testpercent=0):
	X = 4*np.random.rand(nfeature,int(m*(1+testpercent)))-2
	if case == "linear":
		Y = X[0,:] + X[1,:] - 0.25
	if case == "linearwave":
		Y = X[1,:] - np.sin(3.14*X[0,:])
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
	# add noise if requested
	if noise:
		Y = Y + 0.3*np.random.randn(X.shape[1])
	Y = np.maximum(Y,0.0)
	Y = np.round(Y)
	Y = np.minimum(Y,nclass-1)
	Y = np.expand_dims(Y,axis=0)
	# train and validation/test sets
	if testpercent == 0:
		return X,Y
	else:
		return X[:,:m], Y[:,0:m], X[:,m:], Y[:,m:]

if __name__ == "__main__":
	nfeature = 2
	m = 1000
	case = "quadratic"
	nclass = 2
	noise = True
	np.random.seed(10)
	Xtrain,Ytrain,Xvalid,Yvalid = example(nfeature,m,case,nclass,noise,0.2)
	plot_results.plot_results_data((Xtrain,Ytrain),nclass)
	plt.show()