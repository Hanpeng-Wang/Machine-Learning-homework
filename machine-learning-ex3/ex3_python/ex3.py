import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.io as io
import optmodule as opt



def displayData(X, *width):

    m = X.shape[0]
    n = X.shape[1]

    if len(width) == 0:
        example_width = np.round(np.sqrt(n))
    else:
        example_width = width[0]


    example_height = n/example_width
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m/display_rows)

    pad = 1
    rowNum = pad + display_rows * (example_height + pad)
    colNum = pad + display_cols * (example_width + pad)
    display_array = -np.ones((int(rowNum), int(colNum)))

    curr_ex = 0
    for j in range(int(display_rows)):
        for i in range(int(display_cols)):
            if curr_ex >= m:
                break
            
            max_val = np.amax(np.abs(X[curr_ex, :]))
            rowNum = pad + j*(example_height + pad)
            colNum = pad + i * (example_width + pad)
            display_array[int(rowNum):int(example_height + rowNum), int(colNum):int(example_width + colNum)] = \
                        np.transpose(X[curr_ex, :].reshape((int(example_height), int(example_width))))/ max_val
            curr_ex = curr_ex + 1
        

        if curr_ex >= m:
                break 
        
    plt.imshow(display_array)
    plt.show()


## =========== Loading and Visualizing Data =============
mat = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex3/ex3/ex3data1.mat')
print(type(mat))
X =  mat['X']
y = mat['y']
rand_indices = np.random.permutation(X.shape[0])
sel = X[rand_indices[0:100], :]


# displayData(sel)

## ============ Part 2a: Vectorize Logistic Regression ============
theta_t = np.array([-2., -1., 1., 2.])
X_t = np.column_stack((np.ones((5,)), np.arange(1, 16).reshape((5,3))/10))
y_t = (np.array([1,0,1,0,1]) >=0.5)
lambda_t = 3
J, grad = opt.costFunctionReg(theta_t, X_t, y_t, lambda_t)

print("Cost: %f", J)
print("Expected cost: 2.534819")
print("Gradients:" + np.array2string(grad))
print("Expected gradients:[0.146561 -0.548558  0.724722 1.398003]")