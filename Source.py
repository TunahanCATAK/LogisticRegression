import pandas as pd
"""
data = pd.read_csv("C:\\Users\\tr1c4011\\Downloads\\logistic-regression\\Social_Network_Ads.csv")


print(data)
"""

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

size_mapping = {'XL' : 3, 'L': 2, 'M': 1, 'S': 0}
df['size'] = df['size'].map(size_mapping)

print(df)

"""
import numpy as np
# class labels are not ordinal and it doesn't matter which integer number we assign to a particular string-label.
class_mapping = {label: idx for idx,label in
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)


#reverse key-value pairs in the mapping dictionary
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
"""

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)


X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
print(X)

# An even more convenient way to create those dummy features via one-hot encoding is to use the get_dummies.
df = pd.get_dummies(df[['price', 'color', 'size']])
print(df)