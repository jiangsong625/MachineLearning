import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 定义代价函数
def compute_cost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))


# 批量梯度下降法
def gradient_descent(X, Y, theta, alpha, iters ):  # alpha是学习率，iters是迭代次数
    temp = np.matrix(np.zeros(theta.shape))  # 然后将temp变为矩阵[0.,0.]
    parameters = int(theta.ravel().shape[1])  # 参数数量，ravel()的作用是将多维数组变为一维数组
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - Y  # 预测值与样本的差

        for j in range(parameters):
            term = np.multiply(error, X[:, j])  # term是偏导数
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # 更新theta

        theta = temp
        cost[i] = compute_cost(X, Y, theta)

    return theta, cost


# 提取训练样本
df = pd.read_csv('dataFile/ex1data1.txt', names=['population', 'profit'])
df.insert(loc=0, column='ONE', value=1)
# 设置训练值变量x和目标变量y
cols = df.shape[1]
X = df.iloc[:, 0:cols - 1]
Y = df.iloc[:, cols - 1:cols]
# 查看前五行
# print(X.head())
# print(Y.head())
X = np.matrix(X.values)
Y = np.matrix(Y.values)
# 初始化theta
theta = np.matrix(np.array([0, 0]))

# 设置学习率
alpha = 0.01
# 设置迭代次数
iters = 1500

g, cost = gradient_descent(X, Y, theta, alpha, iters)

# # 拟合曲线
# print(g)
# print(cost)
# cost = compute_cost(X, Y, g)
# print(cost)
# X = X[:, 1]
# x = np.linspace(X.min(), X.max(), 100)
# f = g[0, 0] + (g[0, 1] * x)  # f为假设函数
# fig, ax = plt.subplots()
# ax.plot(x, f, 'r', label='Fitted line')
# ax.scatter(df.population, df.profit,  label='Training Data', marker='x')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# plt.show()

# 代价可视化
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'b')
ax.set_xlabel('Iterations')  # 设置x轴表标签
ax.set_ylabel('Cost')  # 设置y轴标签
ax.set_title('Error vs. Training Epoch')  # 设置表头
plt.show()
