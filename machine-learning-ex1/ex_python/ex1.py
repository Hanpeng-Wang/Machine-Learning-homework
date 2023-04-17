import matplotlib.pyplot as plt
import numpy as np
import optmodule as opt


## ======================= define functions =======================
def plotData(x, y, format, size=1):
    plt.plot(x, y, format, ms=size)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex1/ex1/ex1data1.txt', delimiter=',')
X = data[:,0]; y = data[:, 1]
plotData(X, y, 'xr', 10)
m = len(y)
# =================== Part 3: Cost and Gradient descent ===================

temp = np.ones((m,))
X = np.transpose(np.array([np.ones((m,)), data[:,0]]))
theta = np.zeros((2,))

print('Testing the cost function ...')
J = opt.computeCost(X, y, theta)
print('With theta = [0 ; 0], Cost computed = {0:5.5f}'.format(J))
print('Expected cost value (approx) 32.07')

J = opt.computeCost(X, y, np.array([-1, 2]))
print('With theta = [-1 ; 2], Cost computed = {0:5.5f}'.format(J))
print('Expected cost value (approx) 54.24')


print('\nRunning Gradient Descent ...')
iterations = 1500
alpha = 0.01
theta, J_history, theta_history = opt.gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:' + np.array2string(theta))
print('Expected theta values (approx):[-3.6303 1.1664 ]')

## Plot the linear fit
plt.plot(data[:,0], y, 'xr', ms=10, label='Training data')
plt.plot(data[:,0], np.matmul(X, theta), 'b-', label='Linear regression')
plt.legend()
plt.show()


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

Theta1, Theta0 = np.meshgrid(theta1_vals, theta0_vals)
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array((Theta0[i,j], Theta1[i,j]))
        J_vals[i, j] = opt.computeCost(X, y, t)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Theta1, Theta0, J_vals)
plt.show()

fig2, ax2 = plt.subplots(layout='constrained')
ax2.contourf(Theta1, Theta0, J_vals, 15)
ax2.plot(theta[1], theta[0], 'rx', ms=10)
ax2.plot(theta_history[1,:], theta_history[0,:], 'bx', ms=1)
plt.show()