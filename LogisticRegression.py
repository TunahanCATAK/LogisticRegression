import pandas as pd
import numpy as np

def Sigmoid(X, theta):
    h_theta = X @ theta.T
    denumerator = 1.0 + np.exp(-h_theta)
    return 1.0 / denumerator

def ComputeCost(X, y, theta):

    h_theta = Sigmoid(X, theta)

    log_sigmoid = np.log(h_theta)
    log_sigmoid_minus_one = np.log(1 - h_theta)

    mp_log_y = np.multiply(y, log_sigmoid)
    mp_log_y_minus_one = np.multiply(1-y, log_sigmoid_minus_one)

    return (-1*mp_log_y - mp_log_y_minus_one).mean()

def GradientDescent(X, y, theta, alpha, iters, X_test, y_test):
    cost = np.zeros(iters)
    test_cost = np.zeros(iters)
    for i in range(iters):
        total_cost = np.dot(X.T,(Sigmoid(X, theta) - y))
        theta = theta - (((alpha / y.shape[0]) * total_cost).T + 0.01* np.sum(theta))
        cost[i] = ComputeCost(X, y, theta)
        test_cost[i] = ComputeCost(X_test, y_test, theta)
        if i % 10 == 0: # just look at cost every ten loops for debugging
             print(cost[i], test_cost[i])
    return (theta, cost, test_cost)

def PlotCostValues(cost):
    import matplotlib.pyplot as plt

    plt.plot(cost)
    plt.show()


data = pd.read_csv('Social_Network_Ads.csv')
data.columns = ['User ID', 'Gender', 'Age', 'Estimated Salary', 'Class Label']
data['User ID'][:] = 1


X = pd.DataFrame(data.iloc[:,0:4].values)
X.columns = ['Ones', 'Gender', 'Age', 'Estimated Salary']

y = pd.DataFrame(data.iloc[:,4].values)

# when convert from numpy to dateFrame, you should cast your data to numeric.
X['Age'] = X['Age'].apply(pd.to_numeric, errors='ignore')
X['Estimated Salary'] = X['Estimated Salary'].apply(pd.to_numeric, errors='ignore')


# One-hot Encoding for categorized columns.
X = pd.get_dummies(X[['Ones', 'Gender', 'Age', 'Estimated Salary']])

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)

A= X[:,1]
import math
X = np.c_[X, A**2]
A= X[:,2]
X = np.c_[X, A**2]
A= X[:,3]
X = np.c_[X, A**2]
A= X[:,4]
X = np.c_[X, A**2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


theta = np.zeros([1,9])

theta[0,0] = 7.9462611
theta[0,1] = 3.80375914
theta[0,2] = 0.72880819
theta[0,3] = -6.33536582
theta[0,4] = -6.14350963


print(theta)

theta, cost, test_cost = GradientDescent(X_train, y_train, theta, 0.01, 100000, X_test, y_test)
print(theta)
print(ComputeCost(X_test, y_test, theta))

PlotCostValues(cost)
PlotCostValues(test_cost)