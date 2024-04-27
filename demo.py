import numpy as np

z = np.ones((4, 4), dtype=int)
z[:, 2] = 2
print(z)
z = np.matrix(z)
print(z*z)
print(z*z.T)
