import pandas as pd
import numpy as np

df = pd.read_csv('dataFile/ex1data2.txt', names=['size', 'bedrooms', 'price'])
df = (df - df.mean()) / df.std()
print('max:\n', df.max())
print('min:\n', df.min())
