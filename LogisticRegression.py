import pandas as pd

data = pd.read_csv("C:\\Users\\tr1c4011\\Downloads\\logistic-regression\\Social_Network_Ads.csv")
data.columns = ['User ID', 'Gender', 'Age', 'Estimated Salary', 'Class Label']

X = pd.DataFrame(data.iloc[:,1:4].values)
X.columns = ['Gender', 'Age', 'Estimated Salary']

y = pd.DataFrame(data.iloc[:,4].values)

# when convert from numpy to dateFrame, you should cast your data to numeric.
X['Age'] = X['Age'].apply(pd.to_numeric, errors='ignore')
X['Estimated Salary'] = X['Estimated Salary'].apply(pd.to_numeric, errors='ignore')

# One-hot Encoding for categorized columns.
X = pd.get_dummies(X[['Gender', 'Age', 'Estimated Salary']])

print(X)
print(y)