import pandas as pd
import numpy as np
import gradientDescent_single as gradient


if __name__ == '__main__':
    # 读取数据
    X, Y, df = gradient.load_and_preprocess_data('dataFile/ex1data1.txt')
    theta = np.matrix(np.array([0., 0.]))
    alpha = 0.01
    iters = 1500
    theta, cost = gradient.gradient_descent(X, Y, theta, alpha, iters)
    print('cost:', cost)
    print('theta:', theta)
    # 预测
    x = np.matrix([1, 3.5])
    predict = x.dot(theta.T)
    print('predict:', predict*10000)
