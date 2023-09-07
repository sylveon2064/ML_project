#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the dataset
dataset = pd.read_csv("C://Users//tiwar//Downloads//Fuel.csv")
print(dataset)

#declaring the dependent and independent variables
X = dataset.iloc[:,3:12]
y = dataset.iloc[:,12]
print(X)

#one-hot encoding of variables having string input
vehicleclass = pd.get_dummies(X['VEHICLECLASS'],drop_first=True)
transmission = pd.get_dummies(X['TRANSMISSION'],drop_first=True)
fueltype = pd.get_dummies(X['FUELTYPE'],drop_first=True)

#dropping the columns having string input
#here, axis = 1 shows that we are dropping the whole column
X=X.drop('VEHICLECLASS',axis=1)
X=X.drop('TRANSMISSION',axis=1)
X=X.drop('FUELTYPE',axis=1)

#concatenating the encoded columns to the independent variable data
X=pd.concat([X,vehicleclass],axis=1)
X=pd.concat([X,transmission],axis=1)
X=pd.concat([X,fueltype],axis=1)

#converting the dataset into numpy arrays for easier handling of data
X=np.array(X)
y=np.array(y)

#standardising/preprocessing independent variables for better efficiency
for i in range(X.shape[1]-2):
  X[:,i] = (X[:,i] - int(np.mean(X[:,i])))/np.std(X[:,i])
X = np.concatenate((X,np.ones((1067,1))), axis = 1)

#reshaping the dependent variable array for future use
y = y.reshape(len(y),1)

#standardising/preprocessing dependent variable 
y = (y - int(np.mean(y)))/np.std(y)

#function for splitting the data into train set and test set
#we have kept test size as 30% of the original size in order to get better and quicker results 
def split_data_nosklearn(X,y,test_size=0.3,random_state=0):
    np.random.seed(random_state)                 
    indices = np.random.permutation(len(X))      
    data_test_size = int(X.shape[0] * test_size) 
    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test

#implementing the above function
X_train, y_train, X_test, y_test = split_data_nosklearn(X,y)

#defining the function for our linear regression model using gradient descent method
# in this method we keep on decreasing the coefficient i.e theta(or weight) iteratively until we get minimum cost
def linear_regression_model(X, y, learning_rate, iteration):
    m = y.size
    theta = np.zeros((X.shape[1], 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - y))
        d_theta = (1/m)*np.dot(X.T, y_pred - y)
        theta = theta - learning_rate*d_theta
        cost_list.append(cost)# making a cost list for graphical representation
    return theta, cost_list

#no. of iterations has been kept 1000 and learning rate as 0.005 for better accuracy and performance
iteration = 1000
learning_rate = 0.005
theta, cost_list = linear_regression_model(X, y, learning_rate = learning_rate, iteration =
iteration)

#plotting the cost v/s no. of iterations graph to show at what no. of iterations we get minimum cost
x_axis = np.arange(0, iteration)
plt.plot(x_axis, cost_list)
plt.xlabel("No. of iterations")
plt.ylabel("Cost")
plt.show()

# Declaring the predicted array for calculating r-squared error
y_pred = np.dot(X_test, theta)
y_pred.shape # just to make sure our functions are correct

# Importing library from sklearn to get the performance of our model
from sklearn.metrics import r2_score
performance1=r2_score(y_test,y_pred)

# Now we import the sklearn libraries to compare with the model that we have created on our own.

from sklearn.model_selection import train_test_split
#we define new variables of sklearn to segregate the data
A=X
B=y

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(A_train, B_train)

B_pred = regressor.predict(A_test)

from sklearn.metrics import r2_score
performance2=r2_score(B_test,B_pred)

print("Performance of our model =",performance1)
print("Performance of model made by sklearn =",performance2)

