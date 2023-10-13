import numpy as np
from numpy import matlib as matlib
from scipy.optimize import minimize

def linearRegCostFunction(theta, X, y, lamda):
    
    m = len(y)
    errorVec = np.matmul(X, theta) - y
    J1 = 1 / (2 * m) * np.dot(errorVec, errorVec)
    J2 = lamda/(2*m) * (np.dot(theta, theta) - np.square(theta[0]))
    J = J1 + J2


    grad = np.matmul(errorVec, X)/m + lamda/m*theta
    grad[0] = grad[0] - lamda/m*theta[0]

    return J, grad


def trainLinearReg(X, y, lamda):

    initial_theta = np.zeros((X.shape[1],))
    res = minimize(linearRegCostFunction, initial_theta, method='BFGS', jac=True, args=(X, y, lamda), 
                options={'disp': True, "maxiter": 200})
    return res.x


def learningCurve(X, y, Xval, yval, lamda):

    m = len(y)
    error_train = np.zeros((m,))
    error_val = np.zeros((m,))

    # Xval_t = np.column_stack((np.ones((Xval.shape[0],)),Xval))

    for i in range(1, m+1):
       X_t = X[0:i, :]
       Y_t = y[0:i]

       theta = trainLinearReg(X_t, Y_t, lamda)
       
       J, grad = linearRegCostFunction(theta, X_t, Y_t, 0)
       error_train[i-1] = J

       J, grad = linearRegCostFunction(theta, Xval, yval, 0)
       error_val[i-1] = J

    return error_train, error_val


def validationCurve(X, y, Xval, yval):
    
    lamda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros((len(lamda_vec),))
    error_val = np.zeros((len(lamda_vec),))

    i = 0
    for lamda in lamda_vec:
        theta = trainLinearReg(X, y, lamda)

        J, grad = linearRegCostFunction(theta, X, y, 0)
        error_train[i-1] = J

        J, grad = linearRegCostFunction(theta, Xval, yval, 0)
        error_val[i-1] = J
        i = i+1

    return lamda_vec, error_train, error_val


def polyFeatures(X, p):

    X_poly = np.zeros((len(X), p))

    poly = np.arange(1, p+1)
    for i in range(len(X)):
            X_poly[i, :] = np.power(X[i], poly)

    return X_poly


def featureNormalize(X, mu=None, sigma=None):

    shape = np.shape(X)
    
    mu_ = np.mean(X, axis=0) if mu is None else mu
    mu_repmat = matlib.repmat(mu_, shape[0], 1)
    X_norm = X - mu_repmat



    sigma_ = np.std(X_norm, axis=0, ddof=1) if sigma is None else sigma
    sigma_repmat = matlib.repmat(sigma_, shape[0], 1)
    X_norm = np.divide(X_norm,  sigma_repmat)

    return X_norm, mu_, sigma_
       
       



    
