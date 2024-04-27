import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 计算代价函数
def compute_cost(X, Y, theta):
    inner = np.sum(np.power((X * theta.T) - Y, 2))
    return inner / (2 * len(X))


# 梯度下降法
def gradient_descent(X, Y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)  # 存放每次的代价
    for i in range(iters):
        error = (X * theta.T - Y)

        for j in range(parameters):
            term = np.multiply(error, X[:, j])  # 求偏导数
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = compute_cost(X, Y, theta)
    return theta, cost


# 提取文件
df = pd.read_csv('dataFile/ex1data2.txt', names=['square', 'bedrooms', 'price'])
# 特征缩放
df = (df - df.mean()) / df.std()
# 插入常数1
df.insert(loc=0, column='Ones', value=1)
cols = df.shape[1]
# 分配x和y
X = df.iloc[:, 0:cols - 1]
Y = df.iloc[:, cols - 1:cols]
# 将二者转换为矩阵
X = np.matrix(X)
Y = np.matrix(Y)
theta = np.matrix(np.zeros(cols - 1))

# 设置学习率
alpha = 0.01
# 设置迭代次数
iters = 1500
theta, cost = gradient_descent(X, Y, theta, alpha, iters)

# 代价可视化
fig, ax = plt.subplots()  # 详见https://www.runoob.com/matplotlib/matplotlib-subplots.html
ax.plot(np.arange(iters), cost, label='cost',color='r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
plt.grid(color='b', linestyle='--')
plt.show()
