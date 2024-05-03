from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    data = pd.read_csv('dataFile/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    X = data.values[:, :2]
    y = data.values[:, 2]
    return X, y


if __name__ == '__main__':
    X, y = load_and_preprocess_data()
    lr_model = LogisticRegression()
    X_ploy = PolynomialFeatures(degree=1).fit_transform(X)
    lr_model.fit(X_ploy, y)

    print(f'Coefficients: {lr_model.coef_}, Intercept: {lr_model.intercept_}')
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', label='Accepted')
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', label='Rejected')
    print(lr_model.score(X_ploy, y))
    y = lr_model.predict(X_ploy)
    ax.plot(X[:, 0], y, color='red', label='Decision Boundary')
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    ax.legend()
    plt.show()
