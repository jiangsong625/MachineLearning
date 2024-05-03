import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_reg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X.dot(theta.T))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta.T))))
    reg = learning_rate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradient_reg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    params = int(theta.ravel().shape[1])
    grad = np.zeros(params)
    error = sigmoid(X * theta.T) - y
    for i in range(params):
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / m
        else:
            grad[i] = (np.sum(term) / m) + ((learning_rate / m) * theta[0:, i])

    return grad


if __name__ == '__main__':
    # 导入数据
    df = pd.read_csv('dataFile/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    positive = df.loc[df['Accepted'] == 1]
    negative = df.loc[df['Accepted'] == 0]
    # 绘图片
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    # ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    # ax.legend()
    # ax.set_xlabel('Test 1 Score')
    # ax.set_ylabel('Test 2 Score')
    # plt.show()
    # lr_model = LogisticRegression()
    # lr_model.fit(df[['Test 1', 'Test 2']], df['Accepted'])
    # 绘图
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    # ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    # x_min, x_max = ax.get_xlim()
    # print(lr_model.score(df[['Test 1', 'Test 2']], df['Accepted']))

    # degree = 5
    # x1 = df['Test 1']
    # x2 = df['Test 2']
    # df.insert(loc=3, column='Ones', value=1)
    # for i in range(1, degree):
    #     for j in range(0, i):
    #         df['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
    # df.drop('Test 1', axis=1, inplace=True)
    # df.drop('Test 2', axis=1, inplace=True)
    # print(df.head())

    columns = df.shape[1]
    X = df.iloc[:, 0:columns - 1]
    y = df.iloc[:, columns - 1:columns]
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(X.shape[1])
    learning_rate = 1
    cost = cost_reg(theta, X, y, learning_rate)
    grad = gradient_reg(theta, X, y, learning_rate)
    print(cost)
    print(grad)
