# load_mnist.py

import numpy as np
import pandas as pd

def load_mnist(filename,nsample):
	# load data from csv file
	df = pd.read_csv(filename)
	# Y labels are first column and convert to row
	Y = df["label"]
	Y = np.expand_dims(Y,axis=1)
	Y = np.reshape(Y,(1,Y.shape[0]))
	# remove label column
	df=df.drop("label",axis=1)
	# create feature matrix from remaining data and (transpose)
	X = df.values.T
	# samples are rows so take transpose and return
	return X[:,:nsample],Y[:,:nsample]

if __name__ == "__main__":
	filename = "Data/MNIST10K.csv"
	X,Y = load_mnist(filename,1000)
	print("X: {}".format(X[0:10,0:10]))
	print("X.shape: {}".format(X.shape))
	print("Y.shape: {}".format(Y.shape))
