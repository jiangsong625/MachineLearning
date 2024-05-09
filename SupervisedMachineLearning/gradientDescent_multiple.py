import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 计算代价函数
def compute_cost(X, Y, theta):
    inner = np.sum(np.power((X * theta.T) - Y, 2))
    return inner / (2 * len(X))


# 梯度下降法
def gradient_descent(X, Y, theta, alpha, iters):
    """
    使用梯度下降法进行线性回归参数的优化。

    参数:
    X: 训练集的特征矩阵，维度为(n, m)，n为特征数量，m为样本数量。
    Y: 训练集的目标值矩阵，维度为(1, m)。
    theta: 初始参数向量，维度为(n, 1)。
    alpha: 学习率，控制每次迭代更新的步长。
    iters: 迭代次数，决定梯度下降法的停止条件。

    返回值:
    theta: 优化后的参数向量。
    cost: 每次迭代的代价函数值列表。
    """
    cost = np.zeros(iters)  # 初始化代价函数数组

    for i in range(iters):
        # 计算预测值与真实值之间的误差
        error = X.dot(theta.T) - Y

        # 根据误差更新参数向量
        theta -= alpha*(error.T.dot(X))/(len(X))
        cost[i] = compute_cost(X, Y, theta)  # 计算当前迭代的代价函数值

    return theta, cost


# 提取数据
def load_and_preprocess_data(filename):
    """
    从给定的文件中加载数据，并进行预处理。

    参数:
    filename: 字符串，包含房价数据的CSV文件名。

    返回值:
    X: numpy.matrix，处理后的特征矩阵，其中已插入常数项。
    Y: numpy.matrix，处理后的目标变量矩阵。
    data: pandas.DataFrame，原始数据框架，包含特征和目标变量。
    theta: numpy.matrix，初始化的参数矩阵，全为0。
    """
    # 从CSV文件中读取数据，指定列名
    data = pd.read_csv(filename, header=None, names=['square', 'bedrooms', 'price'])

    # 对数据进行特征缩放，使其均值为0，标准差为1
    data = (data - data.mean()) / data.std()

    # 在数据矩阵的开始处插入常数项1，便于后续的模型拟合
    data.insert(loc=0, column='Ones', value=1)

    # 获取处理后数据的列数
    cols = data.shape[1]

    # 分离特征矩阵X和目标变量Y
    X = data.iloc[:, 0:cols - 1]
    Y = data.iloc[:, cols - 1:cols]

    # 将pandas的DataFrame转换为numpy的matrix类型
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)

    # 初始化参数矩阵theta，全为0
    theta = np.matrix(np.zeros(cols - 1))

    return X, Y, data, theta


# 代价可视化
def visualize_cost(cost, iters):
    """
    可视化成本函数随迭代次数的变化。

    参数:
    cost: 一个列表或数组，包含了每个迭代步骤的成本值。
    iters: 整数，表示迭代的次数，即cost列表的长度。

    返回值:
    无返回值，此函数直接展示一个图形化窗口显示成本函数的变化曲线。
    """
    # 创建一个新的图形窗口，并配置两个子图
    fig, ax = plt.subplots()
    # 绘制成本随迭代次数变化的曲线
    ax.plot(np.arange(iters), cost, label='cost', color='r')
    # 设置x轴和y轴的标签
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    # 添加网格线
    plt.grid(color='b', linestyle='--')
    # 显示图形
    plt.show()


# 打印参数
def show_params(X, Y, g):
    """
    Function to display the parameters and the cost associated with those parameters.

    Parameters:
    X (ndarray): The input feature matrix.
    Y (ndarray): The target output vector.
    g (ndarray): The parameters (theta) used in the model.

    Returns:
    None: This function does not return anything; it only prints the parameters and the cost.
    """
    print('theta:', g)  # Displaying the parameters (theta)
    print('cost:', compute_cost(X, Y, g))  # Calculating and displaying the cost associated with the given parameters


if __name__ == '__main__':
    # 设置学习率
    alpha = 0.01
    # 设置迭代次数
    iters = 1500
    X, Y, data, theta = load_and_preprocess_data('dataFile/ex1data2.txt')
    theta, cost = gradient_descent(X, Y, theta, alpha, iters)
    visualize_cost(cost, iters)
    show_params(X, Y, theta)
