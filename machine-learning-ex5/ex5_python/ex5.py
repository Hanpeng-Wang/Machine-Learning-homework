import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import optmodule as opt
from scipy.optimize import minimize


def plotFit(X, mu, sigma, theta, p):
    x_min = np.min(X)
    x_max = np.max(X)

    x = np.arange(x_min-15, x_max+15, 0.05)
    X_poly = opt.polyFeatures(x, p)
    X_poly, mu, sigma = opt.featureNormalize(X_poly, mu, sigma)
    X_poly = np.column_stack((np.ones((X_poly.shape[0],)),X_poly))

    y = np.matmul(X_poly, theta)

    return x, y

## =========== Loading and Visualizing Data =============
mat = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex5/ex5/ex5data1.mat')
print(type(mat))
X =  mat['X']
y = mat['y']
y = np.squeeze(y)

Xval =  mat['Xval']
yval = mat['yval']
Xval_t = np.column_stack((np.ones((Xval.shape[0],)), Xval))
yval = np.squeeze(yval)

Xtest =  mat['Xtest']

# plt.plot(X, y, 'rx', ms=10, linewidth=1.5)
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

##  =========== Part 2: Regularized Linear Regression Cost =============
m = X.shape[0]
X_t = np.column_stack((np.ones((m,)),X))

theta = np.array([1, 1])
J, grad = opt.linearRegCostFunction(theta, X_t, y, 1)

print("Cost: "+ np.array2string(J))
print("Expected cost: 303.993192")
print("Gradients:" + np.array2string(grad))
print("Expected gradients:[-15.303016; 598.250744]")


## ============= Optimizing using scipy  =============

# lamda = 0
# theta = opt.trainLinearReg(X_t, y, lamda)

# plt.plot(X, y, 'rx', ms=10, linewidth=1.5)
# plt.plot(X, np.matmul(X_t, theta), '--', ms=10, linewidth=1.5)
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

## ============= Learning Curve for Linear Regression =============
# lamda = 0
# error_train, error_val = opt.learningCurve(X_t, y, Xval_t, yval, lamda)

# plt.plot(np.arange(0, m), error_train, 'b-', label='Train')
# plt.plot(np.arange(0, m), error_val, 'g-', label='Cross Validation')
# plt.xlabel('Number of training examples')
# plt.ylabel('Error')
# plt.legend()
# plt.show()

## ============= Feature Mapping for Polynomial Regression =============
p = 8
X_poly = opt.polyFeatures(X, p)
X_poly, mu, sigma = opt.featureNormalize(X_poly)
X_poly = np.column_stack((np.ones((X_poly.shape[0],)),X_poly))

X_poly_test = opt.polyFeatures(Xtest, p)
X_poly_test,_,_ = opt.featureNormalize(X_poly_test, mu, sigma)
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0],)),X_poly_test))

X_poly_val = opt.polyFeatures(Xval, p)
X_poly_val,_,_ = opt.featureNormalize(X_poly_val, mu, sigma)
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0],)),X_poly_val))

# lamda = 0
# theta = opt.trainLinearReg(X_poly, y, lamda)


# plt.plot(X, y, 'rx', ms=10, linewidth=1.5)
# plot_x, plot_y = plotFit(X, mu, sigma, theta, p)
# plt.plot(plot_x, plot_y, '--', linewidth=2)
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.title('Polynomial Regression Fit (lambda = ' + str(lamda) + ')')
# plt.show()

# error_train, error_val = opt.learningCurve(X_poly, y, X_poly_val, yval, lamda)
# plt.plot(np.arange(0, m), error_train, 'b-', label='Train')
# plt.plot(np.arange(0, m), error_val, 'g-', label='Cross Validation')
# plt.xlabel('Number of training examples')
# plt.ylabel('Error')
# plt.legend()
# plt.title('Polynomial Regression Learning Curve (lambda = ' + str(lamda) + ')')
# plt.show()

## ============= Validation for Selecting Lambda =============

lamda_vec, error_train, error_val = opt.validationCurve(X_poly, y, X_poly_val, yval)
plt.plot(lamda_vec, error_train, 'b-', label='Train')
plt.plot(lamda_vec, error_val, 'g-', label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

i = 0
for lamda in lamda_vec:
    print(str(lamda), ' ', str(error_train[i]), '   ', str(error_val[i]))
    i = i+1