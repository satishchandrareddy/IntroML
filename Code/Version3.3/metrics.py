# metrics.py

import numpy as np

def f1score(Y,Y_pred):
    # returns f1score, precision, and recall
    # Y = numpy array of actual labels
    # Y_pred = numpy array of predicted labels
    idx_truepositive = np.where((np.absolute(Y-1)<1e-7)&(np.absolute(Y_pred-1)<1e-7))
    idx_actualpositive = np.where(np.absolute(Y-1)<1e-7)
    idx_predpositive = np.where(np.absolute(Y_pred-1)<1e-7)
    precision = np.size(idx_truepositive)/(np.size(idx_predpositive)+1e-7)
    recall = np.size(idx_truepositive)/(np.size(idx_actualpositive)+1e-7)
    f1score = 2*precision*recall/(precision + recall)
    return f1score,precision,recall

def confusion_matrix(Y,Y_pred,nclass):
    # prints confusion matrix
    # Y = 2d numpy array dim(1,# samples) of actual labels
    # Y_pred = 2d numpy array dim(1,# samples) of predicted labels
    # nclass = number of classes
    print("\t\tConfusion Matrix")
    print("\t\tActual")
    strhead = [" \t\t"] + [str(i)+"\t" for i in range(nclass)]
    print("".join(strhead))
    for pred in range(nclass):
        idx_pred = np.where(np.squeeze(np.absolute(Y_pred-pred)<1e-7))
        if pred == 0:
            str_row = ["Predicted  0\t"]
        else:
            str_row = ["           "+str(pred)+"\t"]
        for actual in range(nclass):
            idx_act = np.where(np.squeeze(np.absolute(Y[0,idx_pred]-actual)<1e-7))
            str_row.append(str(np.size(idx_act))+"\t")
        print("".join(str_row))