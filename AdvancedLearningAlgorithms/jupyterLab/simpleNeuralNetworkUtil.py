import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data():
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5
    X[:, 0] = X[:, 0] * (285 - 150) + 150
    Y = np.zeros(len(X))
    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if 175 < t < 260 and 12 < d <= y:
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    return X, Y.reshape(-1, 1)


def plt_output_unit(W, b):
    """ plots a single unit function with 3 inputs """
    steps = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_ = np.linspace(0., 1., steps)
    y_ = np.linspace(0., 1., steps)
    z_ = np.linspace(0., 1., steps)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    d = np.zeros((steps,steps,steps))
    cmap = plt.get_cmap('Blues')
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                v = np.array([x[i,j,k],y[i,j,k],z[i,j,k]])
                d[i,j,k] = tf.keras.activations.sigmoid(np.dot(v,W[:,0])+b).numpy()
    pcm = ax.scatter(x, y, z, c=d, cmap=cmap, alpha = 1 )
    ax.set_xlabel("unit 0");
    ax.set_ylabel("unit 1");
    ax.set_zlabel("unit 2");
    ax.view_init(30, -120)
    ax.figure.colorbar(pcm, ax=ax)
    ax.set_title(f"Layer 2, output unit")

    plt.show()
