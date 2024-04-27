import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_and_preprocess_data(filename):
    """加载数据并进行预处理，添加截距项"""
    df = pd.read_csv(filename, names=['population', 'profit'])
    df.insert(0, 'INTERCEPT', 1)  # 更直观的列名'INTERCEPT'
    return df

def visualize_results(X, y, model_predictions, title):
    """可视化训练数据与模型预测结果"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(X['INTERCEPT'], model_predictions, color='y', label='Prediction')
    ax.scatter(X['population'], y, label='Training Data', alpha=0.6)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()

def calculate_candidate_values():
    """计算一组候选值，示例用途可能为超参数探索"""
    base = np.logspace(-5, -1, num=4)
    candidate_values = np.sort(np.concatenate((base, base * 3)))
    print("Candidate Values:", candidate_values)
    return candidate_values

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
    lr_model.fit(X, y)

    # 预测并绘制结果
    predictions = lr_model.predict(X)
    visualize_results(X, y, predictions, 'Predicted Profit vs. Population Size')

    # 计算并打印候选值
    calculate_candidate_values()