from __future__ import division

import itertools

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Loads data-files
def load_data(fname):
    D = np.loadtxt(fname)
    X = D[:, :-1]
    y = D[:, -1]
    return X, y

# Makes various types of feature vectors
# 'lin' -> linear
# 'quad' -> quadratic
# 'poly' -> polynomial up to degree <degrees>
def make_phi(X, ftype, degrees = None):
    n, d = X.shape
    if ftype == 'lin':
        Phi = np.empty((n, d+1))
        Phi[:, 0] = 1
        Phi[:, 1:] = X
    elif ftype == 'quad':
        Phi_lin = make_phi(X, 'lin')
        ir, ic = np.triu_indices(d)
        Phi_quad = X[:, ir] * X[:, ic]
        Phi = np.hstack([Phi_lin, Phi_quad])
    elif ftype == 'poly':
        Phi_prev = (make_phi(X, 'quad')
                    if degrees <= 3
                    else make_phi(X, 'poly', degrees-1))
        Phi_poly = np.array([[ np.prod(i) for i in itertools.combinations_with_replacement(x, degrees) ] for x in X ])
        Phi = np.hstack([Phi_prev, Phi_poly])
    else:
        raise Exception('Feat type {} not implemented yet'.format(ftype))

    return Phi


# Solves the ridge-regression problem (computes optimal parameters)
def ridgeRegression(X, y, lambda_ = 0):
    n, d = X.shape
    I = np.eye(d); I[0, 0] = 0
    XTX_ridge_inv = la.inv(np.dot(X.T, X) + lambda_ * I)
    return la.multi_dot([XTX_ridge_inv, X.T, y])


# Mean-Squared-Error evaluated on some data, given some parameters beta
def MSE(X, y, beta):
    n, d = X.shape
    diff = np.dot(X, beta) - y
    return np.dot(diff, diff) / n


# Plots the data, and the model fitted to the data if <feats> is also given
def plotDataAndModel(X, y, feats = None, degrees = None):
    n, d = X.shape
    if feats is not None:
        Phi = make_phi(X, feats, degrees)
        beta = ridgeRegression(Phi, y)
    if d == 1:
        plt.plot(X, y, 'o')
        plt.show(block = False)
        if feats != None:
            xhat = np.mgrid[X.min():X.max():.1]
            xhat = xhat.reshape((np.size(xhat), 1))
            xhatfv = make_phi(xhat, feats, degrees)
            yhat = xhatfv.dot(beta)
            plt.plot(xhat, yhat)
    elif d == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X[:, 0], X[:, 1], y, marker = 'o')
        if feats != None:
            xmin = X.min(0)
            xmax = X.max(0)

            x0grid, x1grid = np.mgrid[xmin[0]:xmax[0]:.3, xmin[1]:xmax[1]:.3]

            xdim0, xdim1 = np.shape(x0grid)
            xsize = np.size(x0grid)

            x0hat = x0grid.flatten()
            x1hat = x1grid.flatten()
            x0hat = x0hat.reshape((np.size(x0hat), 1))
            x1hat = x1hat.reshape((np.size(x1hat), 1))
            xhat = np.append(x0hat, x1hat, 1)
            xhatfv = make_phi(xhat, feats, degrees)
            yhat = xhatfv.dot(beta)
            ygrid = yhat.reshape((xdim0, xdim1))
            ax.plot_wireframe(x0grid, x1grid, ygrid)
            ax.auto_scale_xyz([xmin[0], xmax[0]], [xmin[1], xmax[1]], [y.min(), y.max()])
    else:
        raise Exception("Dimensionality of data not handled.")


# Runs cross-validation with one specific value of lambda
def singleCrossValidation(X, y, feats, degrees, nfolds, lambda_):
    n, d = X.shape
    boundaries = np.linspace(0, n, nfolds+1)[1:]
    nind = np.array([ sum(i > boundaries) for i in range(n) ])
    bind = np.equal.outer(range(nfolds), nind)

    loss = np.empty(nfolds)
    for i in range(nfolds):
        X_train, y_train = X[~bind[i]], y[~bind[i]]
        X_test, y_test = X[bind[i]], y[bind[i]]

        Phi_train = make_phi(X_train, feats, degrees)
        Phi_test = make_phi(X_test, feats, degrees)

        beta = ridgeRegression(Phi_train, y_train, lambda_)
        loss[i] = MSE(Phi_test, y_test, beta)

    return loss.mean()

# Runs cross-validation for many values of lambda and plots the errors
def CrossValidation(X, y, feats, degrees, nfolds = 10):
    lambdas = np.outer(10. ** np.arange(-2, 5), [1, 3, 6]).ravel()

    Phi = make_phi(X, feats, degrees)
    betas = np.array([ ridgeRegression(Phi, y, l) for l in lambdas ])
    loss_train = np.array([ MSE(Phi, y, beta) for beta in betas ])
    loss_test = np.array([ singleCrossValidation(X, y, feats, degrees, nfolds, lambda_) for lambda_ in lambdas ])

    #  best lambda
    li = loss_test.argmin()
    lambda_ = lambdas[li]

    plt.figure()

    plt.semilogx(lambdas, loss_train, label='train')
    plt.semilogx(lambdas, loss_test, label='test')
    plt.axvline(lambda_, color='r')

    plt.suptitle('Cross-Validation')
    plt.xlabel('$\lambda$')
    plt.ylabel('Mean Squared Error')

    plt.legend(loc=4)

    return lambda_


############################
# Beginning of the program #
############################


if __name__ == '__main__':
    # Last uncommented line determines data-file to load
    # dataFname = 'datasets/dataLinReg2D.txt'
    # dataFname = 'datasets/dataQuadReg2D.txt'
    dataFname = 'datasets/dataQuadReg2D_noisy.txt'

    # Last uncommented line determines feature type (<degrees> only for 'poly')
    #feats = None
    # feats = 'lin'
    feats = 'quad'
    # feats = 'poly'
    degrees = 3

    X, y = load_data(dataFname);

    # Uncomment to visualize data and fit model
    plotDataAndModel(X, y, feats, degrees)

    # Uncomment to run cross-validation and find best value of lambda
    nfolds = 10
    lambda_ = CrossValidation(X, y, feats, degrees, nfolds)

    print('optimal lambda: {}'.format(lambda_))

    plt.show()
