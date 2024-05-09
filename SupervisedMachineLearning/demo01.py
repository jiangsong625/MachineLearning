import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def load_and_preprocess_data(filename):
    """加载数据并进行预处理，添加截距项"""
    df = pd.read_csv(filename, names=['population', 'profit'])
    # df.insert(0, 'INTERCEPT', 1)  # 更直观的列名'INTERCEPT'
    return df


def visualize_results(X, y, x, model_predictions, title):
    """可视化训练数据与模型预测结果"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, model_predictions, color='r', label='Prediction')
    ax.scatter(X.population, y, label='Training Data', color='cyan', marker='.')
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载并预处理数据
    data_file = 'dataFile/ex1data1.txt'
    df = load_and_preprocess_data(data_file)

    # 分割特征和目标变量
    X = df.drop(columns='profit')
    y = df['profit']

    # 训练线性回归模型
    lr_model = LinearRegression()
    X_poly = PolynomialFeatures(degree=8).fit_transform(X)
    x = np.linspace(X.population.min(), X.population.max(), 100).reshape(-1, 1)
    x_ploy = PolynomialFeatures(degree=8).fit_transform(x)
    lr_model.fit(X_poly, y)
    # 预测并绘制结果
    predictions = lr_model.predict(x_ploy)
    visualize_results(X, y, x, predictions, 'Predicted Profit vs. Population Size')
    print(f'模型得分：{lr_model.score(X_poly, y)}')

