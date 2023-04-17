import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import optmodule as opt


def plotData(x,y,show=True):
    one_index = np.where(y==1)
    zero_index = np.where(y==0)

    plt.plot(x[:, 0][one_index], x[:, 1][one_index], '+r', ms=10, label='y = 1')
    plt.plot(x[:, 0][zero_index], x[:, 1][zero_index], 'oy', ms=10, label='y = 0')
    plt.legend()
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    if show:
        plt.show()


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(np.shape(X1))

    for i in range(1, degree+1):
        for j in range(i+1):
            a = np.power(X1, i-j) * np.power(X2, j)
            out = np.column_stack((out, a))

    return out

def plotDecisionBoundary(theta, X, y, lamda):
    plotData(X, y, False)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U, V = np.meshgrid(u,v)
    z = np.zeros((len(v), len(u)))

    for i in range(len(v)):
        for j in range(len(u)):
            z[i, j] = np.matmul(mapFeature(u[j], v[i]), theta)

    plt.contour(U, V, z, [0])
    # plt.title("lambda = %f", lamda)
    plt.show()
    

def predict(theta, X):
    p = np.around(opt.sigmoid(np.matmul(X, theta)))
    return p

    

## =======================  Plotting =======================
data = np.loadtxt('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex2/ex2/ex2data2.txt', delimiter=',')
X = data[:,0:2]; y = data[:, 2]

# plotData(X,y)
## =========== Regularized Logistic Regression ============
X = mapFeature(X[:,0], X[:,1])

initial_theta = np.zeros((X.shape[1],))

# Set regularization parameter lambda to 1
lamda = 1
cost, grad = opt.costFunctionReg(initial_theta, X, y, lamda)
print('Cost at initial theta (zeros): %f', cost)
print('Expected cost (approx): 0.693')
print("Gradient at initial theta (zeros) - first five values only: " + np.array2string(grad[0:5]))
print('Expected gradients (approx) - first five values only: [0.0085 0.0188 0.0001 0.0503 0.0115]')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10

test_theta = np.ones((X.shape[1],))
cost, grad = opt.costFunctionReg(test_theta, X, y, 10)
print('Cost at test theta (ones): %f', cost)
print('Expected cost (approx): 3.16')
print("Gradient at test theta (ones) - first five values only: " + np.array2string(grad[0:5]))
print('Expected gradients (approx) - first five values only: [0.3460 0.1614 0.1948 0.2269 0.0922]')

## %% ============= Part 2: Regularization and Accuracies =============
initial_theta = np.zeros((X.shape[1],))
lamda = 1.0
res = minimize(opt.costFunctionReg, initial_theta, method='BFGS', jac=True, args=(X, y, lamda), 
                options={'disp': True})

plotDecisionBoundary(res.x, data[:,0:2], y, lamda)


p = opt.predict(res.x, X)
comparison = p == y
accuracy = np.mean(comparison.astype('float64'))
print("Train Accuracy: " + np.array2string(accuracy * 100))
print("Expected accuracy (with lambda = 1): 83.1")