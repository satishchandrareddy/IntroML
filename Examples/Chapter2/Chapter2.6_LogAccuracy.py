#Chapter2.6_LogAccuracy.py

import numpy as np 

# inputs
Y = np.array([[0,1,1,0,1]])
Y_pred = np.array([[1,0,1,0,1]])
# forward propagation
accuracy = np.mean(np.absolute(Y-Y_pred)<1e-7)
print("accuracy: {}".format(accuracy))