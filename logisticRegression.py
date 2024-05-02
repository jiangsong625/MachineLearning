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
    # X = feature_scaling(X)
    # X.insert(0, 'ones', 1)
    X = np.matrix(X)
    y = np.matrix(y).T
    theta = np.matrix(np.zeros(X.shape[1]))
    return X, y, df, theta


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算代价函数
def compute_cost(X, y, theta):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta.T))
    J = (1 / m) * (-np.log(h).T.dot(y) - np.log(1 - h).T.dot(1 - y))
    return J[0, 0]


# 梯度下降算法
def gradient_descent(theta,X, y, alpha, iters):
    m = X.shape[0]
    cost = np.zeros(iters)
    for i in range(iters):
        error = sigmoid(np.dot(X, theta.T)) - y
        theta = theta - np.dot(error.T, X) * (alpha / m)
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


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
    X, y, df, theta = load_and_preprocess_data()
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








