import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 提取文件
df = pd.read_csv('dataFile/ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(df.head())
