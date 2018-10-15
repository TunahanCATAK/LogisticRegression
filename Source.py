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