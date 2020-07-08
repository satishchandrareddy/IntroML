# load_spam.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

def load_spam(train_pct):
	# load data from csv file
	ROOT_DIR = Path(__file__).resolve().parent.parent
	df = pd.read_csv(ROOT_DIR / "Data_Spam/SMSSpamCollection.csv")
	# Get labels from first column and convert to 0 and 1
	Y = np.array(df["label"].map({"ham": 0, "spam": 1}))
	# convert to 1 x nmessages numpy array
	Y = np.reshape(Y,(1,Y.shape[0]))
	print("Number of messages: {}".format(Y.shape[1]))
	print("Number of spam: {}".format(np.sum(Y)))
	print("Number of not- spam: {}".format(np.sum(1-Y)))
	# Get messages from 2nd column
	X = df["message"]
	# Convert to feature matrix using CountVectorizer
	vectorizer = CountVectorizer(decode_error="ignore")
	Xfit = vectorizer.fit_transform(X)
	Xfit_array = Xfit.toarray().T
	print("Xfit_array.shape: {}".format(Xfit_array.shape))
	data_analysis(Xfit_array,Y,10,vectorizer)
	# create training and validation datasets - return also raw messages
	nrows = len(df)
	ntrain = int(train_pct * nrows)
	Xtrain = Xfit_array[:,:ntrain]
	Ytrain = Y[:,:ntrain]
	Xvalid = Xfit_array[:,ntrain:]
	Yvalid = Y[:,ntrain:]
	Xtrain_raw = X[:ntrain].values
	Xvalid_raw = X[ntrain:].values
	print("Xtrain.shape: {}".format(Xtrain.shape))
	print("Ytrain.shape: {}".format(Ytrain.shape))
	print("Xvalid.shape: {}".format(Xvalid.shape))
	print("Yvalid.shape: {}".format(Yvalid.shape))
	print("Xtrain_raw.shape: {}".format(Xtrain_raw.shape))
	print("Xvalid_raw.shape: {}".format(Xvalid_raw.shape))
	return Xtrain,Ytrain,Xvalid,Yvalid,Xtrain_raw,Xvalid_raw

def data_analysis(X,Y,nmostcommon,vectorizer):
	# return indices of most common words in non-spam and spam messages
	# X = dimensions: nwords x nmessages (transpose of output of toarray())
	# Y = label vector (1 x nmessages) vector of 0 and 1
	# vectorizer = instance of CountVectorizer after fit_transform has been performed
	# nmostcommon = number of most common words
	# find index of non-spam messages
	idx_0 = np.where(np.squeeze(np.absolute(Y)<1e-7))
	# find index of spam messages
	idx_1 = np.where(np.squeeze(np.absolute(Y-1)<1e-7))
	# sum over messages and determine idx of most common words in each set
	X0 = np.squeeze(X[:,idx_0])
	X1 = np.squeeze(X[:,idx_1])
	word_idx_0 = np.argsort(-np.sum(X0,axis=1))[:nmostcommon]
	word_idx_1 = np.argsort(-np.sum(X1,axis=1))[:nmostcommon]
	print("Most common words in non-spam messages: {}".format(np.array(vectorizer.get_feature_names())[word_idx_0]))
	print("Most common words in spam messages: {}".format(np.array(vectorizer.get_feature_names())[word_idx_1]))

if __name__ == "__main__":
	Xtrain,Ytrain,Xvalid,Yvalid,Xtrain_raw,Xvalid_raw=load_spam(5000,500)