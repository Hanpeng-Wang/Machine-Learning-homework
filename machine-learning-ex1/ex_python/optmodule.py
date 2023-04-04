import numpy as np
#numpy.matlib is an optional sub-package of numpy that must be imported separately
from numpy import matlib as matlib

def computeCost(X, y, theta):
    m = len(y)
    predictions = np.matmul(X, theta)
    errorVec = predictions - y
    sumSqrErrors = np.dot(errorVec, errorVec)
    J = 1 / (2 * m) * sumSqrErrors
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations,))
    theta_history = np.zeros((len(theta), iterations))

    for iter in range(iterations):
        theta_history[:, iter] = theta
        delta_theta = np.matmul(np.matmul(X, theta) - y, X)/m
        theta = theta - alpha * delta_theta
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history, theta_history



def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    shape = np.shape(X)

    mu_repmat = matlib.repmat(mu, shape[0], 1)
    sigma_repmat = matlib.repmat(sigma, shape[0], 1)

    X_norm = np.divide(X - mu_repmat,  sigma_repmat)
    return X_norm, mu, sigma