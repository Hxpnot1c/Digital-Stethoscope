import pandas as pd

df = pd.read_csv('src/data.csv')
print(df.head())
print(list(df.iloc[:, -1]))