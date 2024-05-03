import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# 特征缩放
def feature_scaling(X):
    return X.apply(lambda column: (column - column.mean()) / column.std())


# 加载数据
def load_and_preprocess_data():
    df = pd.read_csv(filepath_or_buffer='dataFile/ex2data1.txt', names=['text1', 'text2', 'lable'])
    X = df.iloc[:, 0:2]
    y = df.iloc[:, 2]
    X = feature_scaling(X)
    X = np.matrix(X)
    y = np.matrix(y).T
    theta = np.matrix(np.zeros(X.shape[1]))
    b = 0
    return X, y, df, theta, b


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算代价函数
def compute_cost(X, y, theta, b):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta.T) + b)
    J = (1 / m) * (-np.log(h).T.dot(y) - np.log(1 - h).T.dot(1 - y))
    return J[0, 0]

# 计算梯度
def compute_gradient_logistic(X, y, theta, b):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta.T) + b)
    error = h - y
    grad = (1 / m) * (error.T.dot(X))
    b = (1 / m) * np.sum(error)
    return grad, b


# 梯度下降算法
def gradient_descent(theta, b, X, y, alpha, iters):
    m = X.shape[0]
    cost = np.zeros(iters)
    for i in range(iters):
        temp_theta, temp_b = compute_gradient_logistic(X, y, theta, b)
        theta -= temp_theta * alpha
        b -= temp_b * alpha
        cost[i] = compute_cost(X, y, theta, b)
        if i % 100 == 0:
            print(f'迭代次数：{i}代价：{cost[i]}')
    return theta, cost, b


# 代价可视化
def visualize_cost(cost, iters):
    x = np.arange(iters)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, cost, 'r')
    # 添加网格线
    plt.grid(color='b', linestyle='--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Iterations')
    plt.show()


# 主程序
if __name__ == '__main__':
    X, y, df, theta, b = load_and_preprocess_data()
    print(compute_cost(X, y, theta, b))
    theta, b = compute_gradient_logistic(X, y, theta, b)
    theta, cost, b = gradient_descent(theta, b, X, y, 1, 1500)
    print(cost[-1])
    result = X.dot(theta.T) + b
    count = 0
    for i in range(len(result)):
        if result[i] > 0.5:
            result[i] = 1
        else:
            result[i] = 0
        if result[i] == y[i]:
            count += 1
    print(count/len(result))












