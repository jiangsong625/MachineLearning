import numpy
import pandas as pd
import numpy as np
from SupervisedMachineLearning import gradientDescent_single as gradient
import matplotlib.pyplot as plt


# 加载数据
def load_and_preprocess_data(filename):
    # 读取数据
    df = pd.read_csv(filename, header=None, names=['population', 'profit'])
    # 添加一列
    df.insert(0, 'ones', 1)
    # 提取数据
    X = df.iloc[:, 0:2]
    X['population'] = np.sqrt(X['population'])
    Y = df.iloc[:, 2:3]
    # 转换为矩阵
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X, Y, df


if __name__ == '__main__':
    # 读取数据
    X, Y, df = load_and_preprocess_data('dataFile/ex1data1.txt')
    print(type(X))
    theta = np.matrix(np.zeros(2))
    alpha = 0.1
    iters = 1000
    theta, cost = gradient.gradient_descent(X, Y, theta, alpha, iters)
    print('cost:', cost)
    print('theta:', theta)
    # 拟合曲线
    x = np.linspace(0, 30, 100)
    f = theta[0, 0] + theta[0, 1] * numpy.sqrt(x)
    fig, ax = plt.subplots()
    ax.plot(x, f, 'r', label='Fitted line')
    ax.scatter(df.population, df.profit,  label='Training Data', marker='x')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    plt.grid()
    plt.show()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')  # 设置x轴表标签
    ax.set_ylabel('Cost')  # 设置y轴标签
    ax.set_title('Error vs. Training Epoch')  # 设置表头
    plt.show()
