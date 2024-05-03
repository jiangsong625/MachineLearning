import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# 特征缩放
def feature_scaling(X):
    return X.apply(lambda column: (column - column.mean()) / column.std())


# 加载数据
def load_and_preprocess_data():
    df = pd.read_csv(filepath_or_buffer='dataFile/ex2data1.txt', names=['text1', 'text2', 'label'])
    X = df.iloc[:, 0:2]
    y = df.iloc[:, 2]
    X = np.matrix(X)
    y = np.matrix(y).T
    theta = np.matrix(np.zeros(X.shape[1]))
    b = 0
    return X, y, df, theta, b


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算代价函数
def compute_cost(X, y, w, b):
    m = X.shape[0]
    h = sigmoid(np.dot(X, w) + b)
    J = (1 / m) * (-np.log(h).T.dot(y) - np.log(1 - h).T.dot(1 - y))
    return J[0, 0]


# 计算梯度
def compute_gradient_logistic(X, y, w, b, lambda_=None):
    m = X.shape[0]
    h = sigmoid(np.dot(X, w) + b)
    dj_dw = (1 / m) * X.T.dot(h - y)
    dj_db = (1 / m) * np.sum(h - y)
    return dj_dw, dj_db


# 梯度下降算法
def gradient_descent(X, y, w_in, b_in, const_function, gradient_function, alpha, iters, lambda_):
    J_history = []
    w_history = []
    for i in range(iters):
        dj_dw, dj_db = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        J_history.append(const_function(X, y, w_in, b_in))
        if i % 1000 == 0:
            w_history.append(w_in)
            print(f'Iteration {i:4d}: Cost {J_history[-1]:8.2f}')

    return w_in, b_in, J_history, w_history


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


# 画出决策边界
def visualize_boundary(df, model_predictions, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    positive = df[df['label']==1]
    negative = df[df['label']==0]
    ax.scatter(positive['text1'], positive['text2'], marker='x', c='r', label='Admitted')
    ax.scatter(negative['text1'], negative['text2'], marker='o', c='b', label='Not Admitted')
    ax.plot(df['text1'], model_predictions, 'b-')
    ax.set_title(title)
    plt.show()


# 主程序
if __name__ == '__main__':
    X, y, df, w, b = load_and_preprocess_data()
    iteration = 10000
    alpha = 0.001
    initial_w = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
    initial_b = -8
    print(X.dot(initial_w))
    theta, b, J_history, w_history = gradient_descent(
        X, y, initial_w, initial_b, compute_cost,
        compute_gradient_logistic,
        alpha, iteration, 0)
    # visualize_cost(J_history, 10000)
    print(theta, b)
    w1, w2 = theta[0, 0], theta[1, 0]
    model_predictions = -(w1*X[:, 0] + b) / w2
    print(model_predictions)
    visualize_boundary(df, model_predictions, 'Logistic Regression')















