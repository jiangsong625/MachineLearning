import scipy.io as sio
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


tf.random.set_seed(1234)
model = Sequential([
    Dense(units=120, activation='Relu'),
    Dense(units=40,  activation='Relu'),
    Dense(units=6,   activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))