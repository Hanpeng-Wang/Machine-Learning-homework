import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import optmodule as opt


def plotData(x,y,show=True):
    one_index = np.where(y==1)
    zero_index = np.where(y==0)

    plt.plot(x[:, 0][one_index], x[:, 1][one_index], '+r', ms=10, label='Admitted')
    plt.plot(x[:, 0][zero_index], x[:, 1][zero_index], 'oy', ms=10, label='Not admitted')
    plt.legend()
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    if show:
        plt.show()


## =======================  Plotting =======================
data = np.loadtxt('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex2/ex2/ex2data1.txt', delimiter=',')
X = data[:,0:2]; y = data[:, 2]

# print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n")
plotData(X,y)

## ============ Part 2: Compute Cost and Gradient ============
shape = np.shape(X)
m = shape[0]
n = shape[1]
temp = np.ones((m,))
X = np.column_stack((temp, X))

initial_theta = np.zeros((n+1,))
cost, grad = opt.costFunction(initial_theta, X, y)
print("Cost at initial theta (zeros): {0:5.5f}".format(cost))
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros): " + np.array2string(grad))
print("Expected gradients (approx): [-0.1000 -12.0092 -11.2628]")

# Compute and display cost and gradient with non-zero theta

test_theta = np.array([-24, 0.2, 0.2])
cost, grad = opt.costFunction(test_theta, X, y)
print("Cost at test theta: {0:5.5f}".format(cost))
print("Expected cost (approx): 0.218")
print("Gradient at test theta: " + np.array2string(grad))
print("Expected gradients (approx): [0.043  2.566 2.647]")


## ============= Optimizing using scipy  =============
res = minimize(opt.costFunction, initial_theta, method='BFGS', jac=True, args=(X, y), 
                options={'disp': True})

print("Cost at theta found by minimize: {0:5.5f}".format(res.fun))
print("Expected cost (approx): 0.203") 
print("Theta found by minimize:" + np.array2string(res.x))
print("Expected theta (approx):[-25.161  0.206  0.201]") 

plotData(X[:, 1:3],y, False)
plot_x = np.array([np.amin(X[:,1])-2, np.amax(X[:, 1])+2])
plot_y =-(res.x[1]*plot_x + res.x[0])/res.x[2]
plt.plot(plot_x, plot_y, color="b")
plt.show()

## ==============  Predict and Accuracies ==============
prob = opt.sigmoid(np.matmul(np.array([1,45,85]), res.x))
print("For a student with scores 45 and 85, we predict an admission probability of {0:5.5f}".format(prob))
print("Expected value: 0.775 +/- 0.002")


## Compute accuracy on our training set
p = opt.predict(res.x, X)
comparison = p == y
accuracy = np.mean(comparison.astype('float64'))
print("Train Accuracy: " + np.array2string(accuracy * 100))
print("Expected accuracy (approx): 89.0")



