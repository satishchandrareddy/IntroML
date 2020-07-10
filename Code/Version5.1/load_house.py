# load_house.py

import numpy as np
import pandas as pd
from pathlib import Path

def load_house(train_pct,transform=True,standardizeX=True,standardizeY=True):
	# read data file
	root_dir = Path(__file__).resolve().parent.parent
	file_path = root_dir / "Data_House/house.csv"
	data = pd.read_csv(file_path)
	# explanatory variables
	features = ['house-age', 'dist-to-nearest-MRT', 'num-of-stores']
	# response variable
	outcome = ['price-per-unit-area']
	# feature matrix as dataframe
	X_df = data[features]
	# response vector as dataframe
	Y_df = data[outcome]
	# apply transformations
	if transform:
		X0 = np.log(X_df[['dist-to-nearest-MRT']]).T
		X1 = np.sqrt(X_df[['house-age']]).T
		X2 = X_df[['num-of-stores']].values.T
		X = np.concatenate((X0,X1,X2),axis=0)
	else:
		X = X_df.values.T

	Y = Y_df.values.T
	# extract training and validation data
	nrows = len(data)
	ntrain = int(train_pct * nrows)
	Xtrain = X[:,:ntrain]
	Xvalid = X[:,ntrain:]
	Ytrain = Y[:,:ntrain]
	Yvalid = Y[:,ntrain:]
	# standardize training and validation data using training mean and std
	if standardizeX:
	    Xtrain_means = Xtrain.mean(axis=1,keepdims=True)
	    Xtrain_std = Xtrain.std(axis=1,keepdims=True)
	    Xtrain = (Xtrain-Xtrain_means) / Xtrain_std
	    Xvalid = (Xvalid-Xtrain_means) / Xtrain_std
	# standardize Y
	if standardizeY:
		Ytrain_max = np.max(Ytrain)
		Ytrain = Ytrain/Ytrain_max
		Yvalid = Yvalid/Ytrain_max
	# print and return
	print("Xtrain.shape: {} - Ytrain.shape: {}".format(Xtrain.shape,Ytrain.shape))
	print("Xvalid.shape: {} - Yvalid.shape: {}".format(Xvalid.shape,Yvalid.shape))
	return Xtrain,Ytrain,Xvalid,Yvalid

if __name__ == "__main__":
	load_house(0.80,True,True,True)