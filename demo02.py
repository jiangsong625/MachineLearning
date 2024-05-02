import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

print(y)
lr_model = LogisticRegression()
lr_model.fit(X, y)
print(lr_model.intercept_, lr_model.coef_)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
x0 = np.arange(0, 3, 0.1)
x1 = (-lr_model.intercept_ - x0 * lr_model.coef_[0][0]) / lr_model.coef_[0][1]
ax.plot(x0, x1, 'k', lw=1)
plt.show()