#Chapter2.1_LinAccuracy.py

import numpy as np 

# inputs
X = np.array([[1,2,4],[2,5,7]])
Y = np.array([[8,6,10]])
# forward propagation
A = np.array([[4.3170, 7.3828, 10.5844]])
Accuracy = 1/3*np.sum(np.absolute(A-Y))
print("Accuracy: {}".format(Accuracy))