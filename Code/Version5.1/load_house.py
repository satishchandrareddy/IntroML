# load_house.py

import numpy as np
import pandas as pd
from pathlib import Path

def load_house(train_pct,transform=True,standardize=True):
	# read data file
	ROOT_DIR = Path(__file__).resolve().parent.parent
	path = ROOT_DIR / "Data_HousePrices/houseprices.csv"
	data = pd.read_csv(path)
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
		X_df['dist-to-nearest-MRT'] = np.log(X_df['dist-to-nearest-MRT'])
		X_df['house-age'] = np.sqrt(X_df['house-age'])
		Y_df = np.log(Y_df)

	# set seed for reproducibility
	np.random.seed(10)

	# shuffle data
	nrows = len(data)
	shuffled_indices = np.arange(nrows)
	np.random.shuffle(shuffled_indices)
	X_df = X_df.iloc[shuffled_indices].values
	Y_df = Y_df.iloc[shuffled_indices].values

	# extract training and validation data
	ntrain = int(train_pct * nrows)
	Xtrain = X_df[:ntrain,:]
	Xvalid = X_df[ntrain:,:]
	Ytrain = Y_df[:ntrain,:]
	Yvalid = Y_df[ntrain:,:]

	# standardize training and validation data using training mean and std
	if standardize:
	    Xtrain_means = Xtrain.mean(axis=0)
	    Xtrain_std = Xtrain.std(axis=0)
	    Xtrain = (Xtrain-Xtrain_means) / Xtrain_std
	    Xvalid = (Xvalid-Xtrain_means) / Xtrain_std

	# transpose data for training
	Xtrain = Xtrain.T
	Xvalid = Xvalid.T
	Ytrain = Ytrain.T
	Yvalid = Yvalid.T
	print("Xtrain.shape: {} - Ytrain.shape: {}".format(Xtrain.shape,Ytrain.shape))
	print("Xvalid.shape: {} - Yvalid.shape: {}".format(Xvalid.shape,Yvalid.shape))

	return Xtrain,Ytrain,Xvalid,Yvalid