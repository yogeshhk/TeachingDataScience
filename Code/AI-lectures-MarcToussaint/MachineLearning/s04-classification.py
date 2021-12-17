"""
Demo simple logistic regression classification.

See exercise 3 for more.

"""

from __future__ import division
import itertools
import numpy as np
import numpy.random as r
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


def loadData(fname, lfname=None):
    """Loads data-files"""
    if lfname is None:
        D = np.loadtxt(fname)
        X = D[:, 0:-1]
        y = D[:, -1]
    else:
        X = np.loadtxt(fname)
        y = np.loadtxt(lfname)
    return (X, y)


def makeFV(X, feats, degrees=None):
    """
    Makes various types of feature vectors:
    'lin' -> linear
    'quad' -> quadratic
    'poly' -> polynomial up to degree <degrees>

    """
    if feats == 'lin':
        return np.append(np.ones((np.size(X, 0), 1)), X, 1)
    elif feats == 'quad':
        xlin = makeFV(X, 'lin')
        dim = np.size(X, 1)
        ir, ic = np.triu_indices(dim)
        xquad = np.array([np.outer(x, x)[ir, ic] for x in X])
        return np.append(xlin, xquad, 1)
    elif feats == 'poly':
        if degrees <= 3:
            xpoly_prev = makeFV(X, 'quad')
        else:
            xpoly_prev = makeFV(X, 'poly', degrees - 1)
        xpoly = [
            [np.prod(i) for i in itertools.combinations_with_replacement(x, degrees)]
            for x in X
        ]
        return np.append(xpoly_prev, xpoly, 1)

    raise Exception('Feats not implemented yet')


def sigmoid(f):
    return 1 / (1 + np.exp(-f))


def logitRegression(X, y):
    """
    Uses Newton method to solve logistic regression problem
    """
    n, k = X.shape
    XT = X.transpose()
    B = np.zeros(k)
    Eps = np.eye(k) * 1e-9
    i = 0
    while True:
        i += 1
        p = sigmoid(X.dot(B))
        J = XT.dot(p - y)
        W = np.diag(p * (1 - p))
        H = XT.dot(W).dot(X)
        dB = la.inv(H+Eps).dot(J)
        B = B - dB

        print("iter: {}   nll: {}   classificationError: {}".format(
            i, negLogL(X, y, B), classificationError(X, y, B)))
        if la.norm(dB) < 1e-2:
            print(B)
            return B


def plotDataAndModel(X, y, feats=None, degrees=None):
    """
    Plots the data, and the model fitted to the data if <feats> is also given
    """
    n, d = np.shape(X)
    if feats is None:
        labs = np.array(y == 1)
        plt.plot(X[labs][:, 0], X[labs][:, 1], 'ob')
        plt.plot(X[~labs][:, 0], X[~labs][:, 1], 'or')
    else:
        Xfv = makeFV(X, feats, degrees)
        B = logitRegression(Xfv, y)

        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        labs = np.array(y == 1)
        ax.scatter(X[~labs][:, 0], X[~labs][
                   :, 1], 0, marker='o', color='red')
        ax.scatter(X[labs][:, 0], X[labs][
                   :, 1], 1, marker='o', color='blue')

        xmin = X.min(0)
        xmax = X.max(0)

        x0grid, x1grid = np.mgrid[xmin[0]:xmax[0]:.1, xmin[1]:xmax[1]:.1]

        xdim0, xdim1 = np.shape(x0grid)
        xsize = np.size(x0grid)

        x0hat = x0grid.flatten()
        x1hat = x1grid.flatten()
        x0hat = x0hat.reshape((np.size(x0hat), 1))
        x1hat = x1hat.reshape((np.size(x1hat), 1))
        xhat = np.append(x0hat, x1hat, 1)
        xhatfv = makeFV(xhat, feats, degrees)
        yhat = sigmoid(xhatfv.dot(B))
        ygrid = yhat.reshape((xdim0, xdim1))
        ax.plot_wireframe(x0grid, x1grid, ygrid)
        ax.auto_scale_xyz([xmin[0], xmax[0]], [xmin[1], xmax[1]], [0, 1])

        ax = fig.add_subplot(212)
        im = plt.imshow(ygrid.transpose(),
                        interpolation='bilinear',
                        origin='lower',
                        cmap=cm.RdBu,
                        extent=(xmin[0], xmax[0], xmin[1], xmax[1]))
        plt.colorbar(im)
        plt.plot(X[labs][:, 0], X[labs][:, 1], 'ob')
        plt.plot(X[~labs][:, 0], X[~labs][:, 1], 'or')


def negLogL(X, y, B):
    p = sigmoid(X.dot(B))
    nll = -np.sum(np.append(np.log(p)[y == 1], np.log(1 - p)[y == 0]))
    return nll


def classificationError(X, y, B):
    p = sigmoid(X.dot(B))
    l = np.around(p)
    n = y.shape[0]
    return np.sum(np.abs(y - l)) / n


def crossValidation(X, y, feats, degrees, k=10):
    """
    Runs cross-validation with one specific value of lambda
    """
    n, d = np.shape(X)
    boundaries = np.linspace(0, n, k + 1)
    boundaries = boundaries[1:]
    rind = np.arange(n)
    r.shuffle(rind)
    nind = np.array([np.sum(i > boundaries) for i in rind])
    bind = np.array([nind == ki for ki in range(k)])

    Xfv = makeFV(X, feats, degrees)

    nll = np.empty(k)
    ce = np.empty(k)
    for ki in range(k):
        print('ki: ', ki)
        Xtrainfv, Xtestfv = Xfv[~bind[ki]], Xfv[bind[ki]]
        ytrain, ytest = y[~bind[ki]], y[bind[ki]]

        B = logitRegression(Xtrainfv, ytrain)
        nll[ki] = negLogL(Xtestfv, ytest, B)
        ce[ki] = classificationError(Xtestfv, ytest, B)

    # print nll
    # print ce
    return [nll.mean(), ce.mean()]


###############################################################################
def ex2():
    dataFname = 'data2Class.txt'

    feats = None
    feats = 'lin'
    feats = 'quad'

    feats = 'poly'
    degrees = 3 #degree of poly features

    X, y = loadData(dataFname)
    plotDataAndModel(X, y, feats, degrees)
    plt.show()


def ex3():
    feats = None
    feats = 'lin'
    feats = 'quad'
    feats = 'poly'
    degrees = 3

    dataFname = 'digit_pca.txt'
    labelFname = 'digit_label.txt'
    X, y = loadData(dataFname, labelFname)
    mnll, mce = crossValidation(X, y, feats, degrees, 2)
    print('mean neg-log-likelihood: ', mnll)
    print('mean classification error: ', mce)

    plt.show()


if __name__ == '__main__':
    ex2()
    # ex3()
