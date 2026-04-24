from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as m_gaussian

sigma = .1
lambd = sigma**2

# exponential covariance kernel, as requested by the exercise
def ecov_kernel(x, y, l, gamma):
  t1, t2 = np.meshgrid(y, x)
  return np.exp( - ((np.abs(t1 - t2)) / l) ** gamma )

# periodic covariance kernel, to model periodic functions
def sin_kernel(x, y, omega):
  t1, t2 = np.meshgrid(y, x)
  return np.exp(-np.sin(omega*np.pi*(t1-t2)))

# factory, grounds the parameters of a kernel function
def kernel_factory(k, **kwargs):
  return lambda x, y: k(x, y, **kwargs)

# GP update code
def gp(kernel, X, obs = None):
  """
    kernel:     kernel function
    X:          query points
    obs:        tuple of observations
  """

  n_query = X.size
  kxx = kernel(X, X)

  if obs is None:
    return np.zeros(n_query), kxx, np.diag(kxx)

  x, y = obs
  n_obs = x.size

  Ik = np.eye(n_obs)

  # Kernel/Covariance matrices
  kappa = kernel(X, x)
  K = kernel(x, x)

  post_mean = kappa.dot(np.linalg.inv(K + lambd*Ik)).dot(y).flatten()
  post_cov = kxx - kappa.dot(np.linalg.inv(K + lambd*Ik)).dot(kappa.transpose())
  post_std = np.sqrt(np.diagonal(post_cov))
  return post_mean, post_cov, post_std

def ex1():
  x = np.column_stack((np.array([-.5, .5]),))
  y = np.column_stack((np.array([.3, -.1]),))

  # TODO - uncomment to change observed datapoints
  # x = np.column_stack((np.array([-.5, .5, 0]),))
  # y = np.column_stack((np.array([.3, -.1, 1]),))

  # TODO - uncomment to change observed datapoints
  # x = np.column_stack((np.linspace(-.9, .9, 8),))
  # y = np.column_stack((np.sin(np.pi*x),))

  # observations
  obs = (x, y)

  draw = 3
  n_query = 100
  X = np.linspace(-1, 1, n_query)
  I = np.eye(n_query)

  # Kernel functions
  k1 = kernel_factory(ecov_kernel, l = .2, gamma = 2)
  k2 = kernel_factory(ecov_kernel, l = .2, gamma = 1)

  # Posterior GP parameters
  prior_m, prior_cov1, prior_std1 = gp(k1, X)
  prior_m, prior_cov2, prior_std2 = gp(k2, X)
  post_m1, post_cov1, post_std1 = gp(k1, X, obs)
  post_m2, post_cov2, post_std2 = gp(k2, X, obs)

  # Sampling from the GPs
  prior_sample1 = m_gaussian(prior_m, prior_cov1, draw)
  post_sample1 = m_gaussian(post_m1, post_cov1, draw)
  post_sample1_obs = m_gaussian(post_m1, post_cov1 + I*sigma**2, 1)

  prior_sample2 = m_gaussian(prior_m, prior_cov2, draw)
  post_sample2 = m_gaussian(post_m2, post_cov2, draw)
  post_sample2_obs = m_gaussian(post_m2, post_cov2 + I*sigma**2, 1)

  # Plotting
  plt.clf()

  plt.subplot(2, 2, 1)
  plt.plot(X, prior_m, 'k', linewidth = 3)
  plt.fill_between(X, prior_m+prior_std1, prior_m-prior_std1, alpha = .5)
  for ps in prior_sample1:
    plt.plot(X, ps)
  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  plt.subplot(2, 2, 2)
  plt.plot(X, prior_m, 'k', linewidth = 3)
  plt.fill_between(X, prior_m+prior_std2, prior_m-prior_std2, alpha = .5)
  for ps in prior_sample2:
    plt.plot(X, ps)
  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  plt.subplot(2, 2, 3)
  plt.plot(X, post_m1, 'k', linewidth = 3)
  plt.plot(x, y, 'kx', markersize = 7.5, markeredgewidth = 2)
  plt.fill_between(X, post_m1+post_std1, post_m1-post_std1, alpha = .5)
  for ps in post_sample1:
    plt.plot(X, ps)
  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  plt.subplot(2, 2, 4)
  plt.plot(X, post_m2, 'k', linewidth = 3)
  plt.plot(x, y, 'kx', markersize = 7.5, markeredgewidth = 2)
  plt.fill_between(X, post_m2+post_std2, post_m2-post_std2, alpha = .5)
  for ps in post_sample2:
    plt.plot(X, ps)
  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

#  plt.subplot(3, 2, 5)
#  plt.plot(X, post_m1, 'k', linewidth = 3)
#  plt.plot(x, y, 'kx', markersize = 7.5, markeredgewidth = 2)
#  plt.fill_between(X, post_m1+post_std1, post_m1-post_std1, alpha = .5)
#  for ps in post_sample1_obs:
#    plt.plot(X, ps, 'r.')
#  plt.xlim(-1, 1)
#  plt.ylim(-2, 2)

#  plt.subplot(3, 2, 6)
#  plt.plot(X, post_m2, 'k', linewidth = 3)
#  plt.plot(x, y, 'kx', markersize = 7.5, markeredgewidth = 2)
#  plt.fill_between(X, post_m2+post_std2, post_m2-post_std2, alpha = .5)
#  for ps in post_sample2_obs:
#    plt.plot(X, ps, 'r.')
#  plt.xlim(-1, 1)
#  plt.ylim(-2, 2)
  plt.show()

if __name__ == '__main__':
  ex1()

