import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 定义代价函数
def compute_cost(X, Y, theta):
    inner = np.power(X.dot(theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))


# 批量梯度下降法
def gradient_descent(X, Y, theta, alpha, iters ):  # alpha是学习率，iters是迭代次数
    cost = np.zeros(iters)
    for i in range(iters):
        error = X.dot(theta.T) - Y  # 使用dot进行矩阵乘法，得到预测值与样本的差

        # 使用向量化方式更新theta，避免循环
        update = (alpha / len(X)) * error.T.dot(X)
        # 更新theta
        theta = theta - update
        cost[i] = compute_cost(X, Y, theta)

    return theta, cost


# 提取训练样本
def load_and_preprocess_data(filename):
    data = pd.read_csv(filename, names=['population', 'profit'])
    data.insert(loc=0, column='ONE', value=1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    Y = data.iloc[:, cols - 1:cols]
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X, Y, data


# 拟合曲线
def visualize_results(df, g):
    # 拟合曲线
    x = np.linspace(df.population.min(), df.population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)  # f为假设函数
    fig, ax = plt.subplots()
    ax.plot(x, f, 'r', label='Fitted line')
    ax.scatter(df.population, df.profit,  label='Training Data', marker='x')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    plt.show()


# 代价可视化
def visualize_cost(cost, iters):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')  # 设置x轴表标签
    ax.set_ylabel('Cost')  # 设置y轴标签
    ax.set_title('Error vs. Training Epoch')  # 设置表头
    plt.show()


# 显示参数
def show_params(X, Y, g):
    # 打印参数
    print('theta=', g)
    # 打印均方误差 MSE(Mean Squared error)
    inner = np.power((X * g.T) - Y, 2)
    print('MSE:', np.mean(inner))
    # 打印代价
    print('Cost:', compute_cost(X, Y, g))


# 主程序
if __name__ == '__main__':
    X, Y, df = load_and_preprocess_data('dataFile/ex1data1.txt')
    # 初始化theta
    theta = np.matrix(np.array([0, 0]))
    # 设置学习率
    alpha = 0.01
    # 设置迭代次数
    iters = 2000
    # 拟合模型
    g, cost = gradient_descent(X, Y, theta, alpha, iters)
    visualize_results(df, g)
    visualize_cost(cost, iters)
    show_params(X, Y, g)


