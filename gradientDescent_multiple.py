import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 计算代价函数
def compute_cost(X, Y, theta):
    inner = np.sum(np.power((X * theta.T) - Y, 2))
    return inner / (2 * len(X))


# 梯度下降法
def gradient_descent(X, Y, theta, alpha, iters):
    cost = np.zeros(iters)  # 存放每次的代价
    for i in range(iters):
        error = X.dot(theta.T) - Y

        theta -= alpha*(error.T.dot(X))/(len(X))
        cost[i] = compute_cost(X, Y, theta)

    return theta, cost


# 提取数据
def load_and_preprocess_data(filename):
    # 提取文件
    data = pd.read_csv(filename, header=None, names=['square', 'bedrooms', 'price'])
    # 特征缩放
    data = (data - data.mean()) / data.std()
    # 插入常数1
    data.insert(loc=0, column='Ones', value=1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    Y = data.iloc[:, cols - 1:cols]
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    theta = np.matrix(np.zeros(cols - 1))
    return X, Y, data, theta


# 代价可视化
def visualize_cost(cost, iters):
    fig, ax = plt.subplots()  # 详见https://www.runoob.com/matplotlib/matplotlib-subplots.html
    ax.plot(np.arange(iters), cost, label='cost',color='r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    plt.grid(color='b', linestyle='--')
    plt.show()


# 打印参数
def show_params(X, Y, g):
    print('theta:', g)
    print('cost:', compute_cost(X, Y, g))


if __name__ == '__main__':
    # 设置学习率
    alpha = 0.01
    # 设置迭代次数
    iters = 1500
    X, Y, data, theta = load_and_preprocess_data('dataFile/ex1data2.txt')
    theta, cost = gradient_descent(X, Y, theta, alpha, iters)
    visualize_cost(cost, iters)
    show_params(X, Y, theta)
