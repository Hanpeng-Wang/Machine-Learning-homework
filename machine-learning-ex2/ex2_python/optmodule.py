import numpy as np

def sigmoid(z):
    h = 1 / (np.exp(-z) + 1)
    return h

def predict(theta, X):
    p = np.around(sigmoid(np.matmul(X, theta)))
    return p


def costFunction(theta, X, y):
    m = len(y)
  
    h = sigmoid(np.matmul(X, theta))

    J1 = np.matmul(y, np.log(h))
    J2 = np.matmul(1-y, np.log(1 - h))
    J = -1/m * (J1 + J2)
    grad = np.matmul(h - y, X)/m

    return J, grad

def costFunctionReg(theta, X, y, lamda):

    m = X.shape[0]
    J, grad = costFunction(theta, X, y)
    J = J + lamda/(2*m)*(np.sum(np.square(theta)) - np.square(theta[0]))
    grad = grad + lamda/m*theta
    grad[0] = grad[0] - lamda/m*theta[0]

    return J, grad

    
    
    