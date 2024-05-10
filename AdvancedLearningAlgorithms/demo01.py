import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
import logging


logging.getLogger("tensorflow").setLevel(logging.ERROR)  # 暂时取消警告


X_train = np.array([[1.], [2.]], dtype=np.float32)
Y_train = np.array([[300.], [500.]], dtype=np.float32)
linear_model = Dense(units=1, activation='linear', name='linear_model')
print(linear_model.weights)


