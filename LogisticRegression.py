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

def GradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        total_cost = np.dot(X.T,(Sigmoid(X, theta) - y))
        theta = theta - ((alpha / y.shape[0]) * total_cost).T
        cost = ComputeCost(X, y, theta)
        if i % 10 == 0: # just look at cost every ten loops for debugging
             print(cost)
    return (theta, cost)


data = pd.read_csv('Social_Network_Ads.csv')
data.columns = ['User ID', 'Gender', 'Age', 'Estimated Salary', 'Class Label']

X = pd.DataFrame(data.iloc[:,1:4].values)
X.columns = ['Gender', 'Age', 'Estimated Salary']

y = pd.DataFrame(data.iloc[:,4].values)

# when convert from numpy to dateFrame, you should cast your data to numeric.
X['Age'] = X['Age'].apply(pd.to_numeric, errors='ignore')
X['Estimated Salary'] = X['Estimated Salary'].apply(pd.to_numeric, errors='ignore')

# One-hot Encoding for categorized columns.
X = pd.get_dummies(X[['Gender', 'Age', 'Estimated Salary']])

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


theta = np.ones([1,4])
"""
theta[0,0] = 9.38338121
theta[0,1] = 4.62322883
theta[0,2] = -7.52121024
theta[0,3] = -7.21505842
"""

print(theta)

theta, cost = GradientDescent(X_train, y_train, theta, 0.01, 10000)
print(theta)


