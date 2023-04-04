import matplotlib.pyplot as plt
import numpy as np
import optmodule as opt




data = np.loadtxt('D:\Machine-Learning-homework\machine-learning-ex1\ex1\ex1data2.txt', delimiter=',')
X = data[:,0:2]; y = data[:, 2]
m = len(y)
## ================ normalize feature ================

X, mu, sigma = opt.featureNormalize(X)
temp = np.ones((m,))
X = np.column_stack((temp, X))


## ================ Part 2: Gradient Descent ================
print('Running gradient descent ...\n')
alpha = 0.01
num_iters = 1000
theta = np.zeros((3,))
theta, J_history, theta_history = opt.gradientDescent(X, y, theta, alpha, num_iters)

print('Theta computed from gradient descent:'+np.array2string(theta))
print('Cost computed from gradient descent:{0:5.5f}'.format(opt.computeCost(X,y,theta)))
## Plot the convergence graph
plt.plot(np.arange(num_iters), J_history, 'b-', lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()



## ================ Part 3: Normal Equations ================
print('\nSolving with normal equations...')
XT = np.transpose(X)
XTX = np.matmul(XT, X)
theta_normal = np.matmul(np.matmul(np.linalg.inv(XTX), XT), y)
print('Theta computed from the normal equations:'+np.array2string(theta_normal))
print('Cost computed from gradient descent:{0:5.5f}'.format(opt.computeCost(X,y,theta_normal)))