# import tensorflow.keras
import numpy as np


X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
print(X_train[1].reshape(1, 1))
Xt = np.tile(X_train, (1, 1, 100))
print(Xt)