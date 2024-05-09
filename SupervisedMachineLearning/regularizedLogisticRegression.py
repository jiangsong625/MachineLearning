import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost_reg(w, b, X, y, learning_rate):
    m = X.shape[0]
    cost = (-1 / m) * (np.dot(y.T, np.log(sigmoid(np.dot(X, w.T) + b))) + np.dot((1 - y), np.log(1 - sigmoid(np.dot(X, w.T) + b))))
    reg = (learning_rate / (2 * m)) * np.sum(w[1:] ** 2)
    return cost + reg


# 梯度
def compute_gradient(w, b, X, y):
    m = X.shape[0]
    dj_dw = (1 / m) * (np.dot(X.T, sigmoid(np.dot(X, w.T)+b) - y))
    dj_db = (1 / m) * np.sum(sigmoid(np.dot(X, w.T)+b) - y)
    return dj_dw, dj_db


def compute_gradient_reg(w, b, X, y, learning_rate):
    m = X.shape[0]
    dj_dw, dj_db = compute_gradient(w, b, X, y)
    dj_dw += (learning_rate / m) * w
    return dj_dw, dj_db


def gradient_descent_reg(X, y, w_in, b_in, cost_function, gradient_function, alpha, iters, lambda_):
    J_history = []
    w_history = []
    for i in range(iters):
        dj_dw, dj_db = gradient_function(w_in, b_in, X, y, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        J_history.append(cost_function(w_in, b_in, X, y, lambda_))
        if i % 100000 == 0:
            w_history.append(w_in)
            print(f'Iteration {i:4d}: Cost {J_history[-1]:8.2f}')

    return w_in, b_in, J_history, w_history


def load_and_preprocess_data():
    df = pd.read_csv('dataFile/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    data = df.copy()
    # 特征映射
    degree = 6
    for i in range(1, degree+1):
        for j in range(i + 1):
            df['F' + str(i - j) + str(j)] = (df['Test 1'] ** (i - j)) * (df['Test 2'] ** j)
    df.drop(['Test 1', 'Test 2'], axis=1, inplace=True)
    X = df.drop('Accepted', axis=1)
    y = df['Accepted']
    return data, df, X, y


if __name__ == '__main__':
    # 导入数据
    data, df, X, y = load_and_preprocess_data()
    initial_w = np.random.rand(X.shape[1]) - 0.5
    initial_b = 0.5
    iterations = 300000
    alpha = 0.1
    lambda_ = 0.01
    # print(compute_cost_reg(initial_w, initial_b, X, y, 1))
    # print(compute_gradient_reg(initial_w, initial_b, X, y, 1))
    w, b, J_history, w_history = gradient_descent_reg(
        X.values, y.values, initial_w, initial_b,
        compute_cost_reg, compute_gradient_reg,
        alpha, iterations, lambda_)
    # 预测
    predictions = sigmoid(np.dot(X, w.T) + b)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    # 显示模型准确性，约为0.83
    print(np.sum((y-predictions) == 0) / len(predictions))





