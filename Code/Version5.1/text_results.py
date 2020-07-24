# plot_results.py

import numpy as np

def text_results(Y,Y_pred,X_raw):
    # determine indices for false positive (Y = 0 and Y_pred = 1)
    idx_falsepositive = np.where(np.squeeze(Y<1e-7)&np.squeeze(np.absolute(Y_pred-1)<1e-7))
    # determine indices for false negative (Y = 1 and Y_pred = 0)
    idx_falsenegative = np.where(np.squeeze(np.absolute(Y-1)<1e-7)&np.squeeze(Y_pred<1e-7))
    # output results
    print("--------------------------------------------------------------------")
    print("False Positive messages - Actual = not spam - Predicted = spam")
    for message in X_raw[idx_falsepositive]:
        print("{}".format(message))
    print("--------------------------------------------------------------------")
    print("False Negative messages - Actual = spam - Predicted = not spam")
    for message in X_raw[idx_falsenegative]:
        print("{}".format(message))