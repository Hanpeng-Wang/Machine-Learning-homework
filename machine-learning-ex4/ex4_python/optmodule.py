import numpy as np


def sigmoid(z):
    h = 1.0 / (np.exp(-z) + 1.0)
    return h

def sigmoidGradient(z):
    g = np.multiply(sigmoid(z), (1-sigmoid(z)))
    return g


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init 


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],(num_labels, (hidden_layer_size + 1)))

    h = np.identity(num_labels)
    y = h[y-1,:]

    m = X.shape[0]
    a1 = np.column_stack((np.ones((m,)),X))
    z2 = np.matmul(a1, Theta1.transpose())
    a2 = sigmoid(z2)

    a2 = np.column_stack((np.ones((a2.shape[0],)),a2))
    z3 = np.matmul(a2, Theta2.transpose())
    a3 = sigmoid(z3)

    J1 = np.sum(np.multiply(y, np.log(a3)))
    J2 = np.sum(np.multiply(1-y, np.log(1-a3)))
    J = -1/m * (J1 + J2)

    regularized = 1/(2*m)*(np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))
    J = J + lamda * regularized


    delta3 = a3 - y
    delta2 = np.multiply(np.matmul(delta3, Theta2)[:,1:], sigmoidGradient(z2))

    Theta1_grad = (1/m) * np.matmul(np.transpose(delta2), a1) + (lamda/m) * Theta1
    Theta2_grad = (1/m) * np.matmul(np.transpose(delta3), a2) + (lamda/m) * Theta2

    Theta1_grad[:, 0] = Theta1_grad[:, 0] - (lamda/m) * Theta1[:, 0]
    Theta2_grad[:, 0] = Theta2_grad[:, 0] - (lamda/m) * Theta2[:, 0]

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    

    return J,  grad

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