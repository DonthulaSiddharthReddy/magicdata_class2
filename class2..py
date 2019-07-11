# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize as op

#-----data-visuavalization-------------#
# load the data from the file
data =pd.read_csv("/home/siddhu/Miniproject/datasets/magic/dataset.csv", header=None)

# X = feature values, all the columns except the last column
X = data.iloc[:,:-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 'g']

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 'h']

# plots
plt.scatter(admitted.iloc[0:500, 0], admitted.iloc[0:500, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[0:500, 0], not_admitted.iloc[0:500, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()

#-----------creating matrics-----------------------#

data1=np.array(data)
Y=data1[1:,-1]#output set

def convert(Y):#converting g to 1 and h to zero
	y=np.zeros((1,np.size(Y)))
	for i in range(np.size(Y)):
		if Y[i]=='g':
			y[0,i]=1
		else:
			y[0,i]=0
	return y

def deconvert(Y):#converting 1 to g and 0 to h
	y=np.zeros((1,np.size(Y))).astype('str')
	for i in range(np.size(Y)):
		if Y[0,i]==1:
			y[0,i]="g"
		else:
			y[0,i]="h"
	return y


R= np.c_[np.ones((X.shape[0], 1)), X]
y=convert(Y)#converted matrix 
y = y[:, np.newaxis]
X1=R[1:,:]#input set

theta = np.zeros((X1.shape[1], 1))

X1=X1.astype('float64')

#---------------functions--------------------------#

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def input(theta, x):
    # Computes the weighted sum of inputs
    return (np.dot(x, theta))

def output(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    cost = -(1 / m) * np.sum(y * np.log(output(theta, x)) + (1 - y) * np.log(1 - output(theta, x)))
    return cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(input(theta,   x)) - y)

def fit(x, y, theta):
    # minimizing fuction
	weights = op.fmin_tnc(func=cost_function, x0=theta,
	                  fprime=gradient,args=(x, y.flatten()))
	return weights[0]
#-----------------------------------------final parameters and acurracy---------------------------------------#
parameters = fit(X1, y, theta)

m=np.size(y)
final_theta=parameters.reshape(-1,1)
prediction=output(final_theta, X1)
predict=np.array([(0)])
for i in prediction:
	if i<0.5:
		predict=np.hstack((predict,np.array([(0)])))
	else:
		predict=np.hstack((predict,np.array([(1)])))

predict=predict.reshape(1,(m+1))

final_set=predict[:,1:]

f=abs(final_set-y)
accuracy=100-(((np.sum(f))/m)*100)
print(accuracy,"%","accuracy")


#------------------------------------------final data visuvalization-------------------------------------------------#
x_values = [np.min(X1[:, 1]), np.max(X1[:, 2])]
#y_values = (- (parameters[0] + np.dot(parameters[1], x_values)+np.dot(parameters[10], x_values)+np.dot(parameters[3], x_values)
#				+np.dot(parameters[4], x_values)+np.dot(parameters[5], x_values)+np.dot(parameters[6], x_values)
#				+np.dot(parameters[7], x_values)+np.dot(parameters[8], x_values)+np.dot(parameters[9], x_values)) / parameters[2])
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
print(x_values)
print(y_values)


# X = feature values, all the columns except the last column
X = data.iloc[:,:-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 'g']

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 'h']




plt.plot(x_values, y_values, label='Decision Boundary')
plt.scatter(admitted.iloc[0:500, 0], admitted.iloc[0:500, 9], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[0:500, 0], not_admitted.iloc[0:500, 9], s=10, label='Not Admitted')
#plt.xlim(1, 50)
plt.legend()
#plt.show()

#----------------------------------------final output----------------------------------------------------------------------#

final_output=deconvert(final_set)
print(final_output)
    
