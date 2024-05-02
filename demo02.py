import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd


df = pd.read_csv(filepath_or_buffer='dataFile/ex2data1.txt', names=['text1', 'text2', 'lable'])
X = df.iloc[:, 0:2]
y = df.iloc[:, 2]
X = np.array(X)
y = np.array(y).reshape(y.shape[0],)
lr_model = LogisticRegression()
lr_model.fit(X, y)
print(lr_model.coef_)
x0 = np.linspace(df['text1'].min(), df['text1'].max(), 100)
x1 = (-1 / lr_model.coef_[0][1]) * (lr_model.coef_[0][0] * x0 + lr_model.intercept_)
# 打印图像
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['text1'], df['text2'], c=df['lable'], s=40, cmap=plt.cm.Spectral)
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')
ax.plot(x0, x1, color='red', linewidth=3)
plt.show()
