import numpy as np
import numpy.random as rnd

from scipy.stats import logistic

data = np.loadtxt('datasets/data2Class_adjusted.txt')
X = data[:, :-1]
Y = data[:, -1]

# Center data (forgot to do it directly in dataset file)
X -= X.mean(axis=0)
X[:, 0] = 1

def forward(x0, w):
    """ NN forward propagation algorithm.

    Inputs (x0, w):
        x0: Network input values
        w: Tuple of weights, one for each layer
    
    Outputs (zL, xl_list):
        zL: Output neuron discriminative function (NOT passed through sigmoid)
        xl_list: list of activation values for each layer (YES passed through sigmoids)
    """
    xl = x0
    xl_list = [xl]
    for wl in w:
        zl = np.dot(wl, xl)
        xl = logistic.cdf(zl)
        xl_list.append(xl)
    return np.asscalar(zl), xl_list

def backward(dz, x0, w):
    """ NN backward propagation algorithm.

    Inputs (dz, x0, w):
        dz: Output neuron loss gradient
        x0: Network input values
        w: Tuple of weights, one for each layer

    Outputs (dw_list):
        dw_list: Tuple of weight gradients
    """
    zl, xl_list = forward(x0, w)
    dw_list = []
    for xl, wl in zip(reversed(xl_list[:-1]), reversed(w)):
        dw = np.outer(dz, xl.T)
        dz = np.dot(dz, wl) * xl * (1 - xl)
        dw_list.append(dw)
    return reversed(dw_list)

def gradient_descent_single_step(X, Y, w):
    """ Single gradient backprop step.  Not pretty, but works; too lazy to improve clarity. """
    dw_lists = []
    loss_list  = []
    for x, y in zip(X, Y):
        f = forward(x, w)[0]
        loss = np.maximum(0, 1 - y * f)
        loss_list.append(loss)
        if loss > 0:
            dz = np.array([-y])
            dw_lists.append(backward(dz, x, w))
    dw = (np.zeros((100, 3)), np.zeros((1, 100)))
    for dw_list in dw_lists:
        for dwl, _dwl in zip(dw, dw_list):
            dwl += _dwl
    return np.mean(loss_list), dw

# weight initializations (not too important to select range precisely in this simple instance
w = (
    rnd.uniform(-1, 1, (100, 3)),  # hidden layer weights
    rnd.uniform(-1, 1, (1, 100)),  # output layer weights
    #np.zeros((100, 3)),  # hidden layer weights
    #np.zeros((1, 100)),  # output layer weights
)

alpha = .05  # step size
# niter = 150
niter = 50
mean_loss_list = [None] * niter
for i in range(niter):
    mean_loss, dw = gradient_descent_single_step(X, Y, w)
    mean_loss_list[i] = mean_loss
    print('{}) mean_loss: {}'.format(i, mean_loss))
    w = tuple(wl - alpha * dwl for wl, dwl in zip(w, dw))

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

# plot loss improvement during backprop
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mean_loss_list)
ax.set_xlabel('iter no.')
ax.set_ylabel('mean loss')

# compute network value over a grid
xlim = 2
ax_xlim = xlim + .5

Xgrid = np.linspace(-xlim, xlim, 100)
X0, X1 = np.meshgrid(Xgrid, Xgrid)
XX = np.array([[1, x0, x1] for x0, x1 in zip(X0.ravel(), X1.ravel())])
ff = np.array([logistic.cdf(forward(xx, w)[0]) for xx in XX])
F = ff.reshape(X0.shape)
Z = (F > 0).astype(int)

# plot network value over the grid
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[Y == -1, 1], X[Y == -1, 2], ff.min(), c='b')
ax.scatter(X[Y == 1, 1], X[Y == 1, 2], ff.min(), c='r')
ax.plot_surface(X0, X1, F, rstride=8, cstride=8, alpha=0.3)

#cset = ax.contourf(X0, X1, F, zdir='x', offset=-xlim, cmap=cm.coolwarm)
#cset = ax.contourf(X0, X1, F, zdir='y', offset=xlim, cmap=cm.coolwarm)
cset = ax.contourf(X0, X1, F, zdir='z', offset=ff.min(), cmap=cm.coolwarm)

ax.set_xlabel('x0')
ax.set_xlim(-ax_xlim, ax_xlim)
ax.set_ylabel('x1')
ax.set_ylim(-ax_xlim, ax_xlim)
ax.set_zlabel('f')
ax.set_zlim(ff.min(), ff.max())

plt.show()
