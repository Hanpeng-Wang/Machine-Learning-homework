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


def predict(Theta1, Theta2, X):
    # feedforward predict
    m = X.shape[0]
    X = np.column_stack((np.ones((m,)),X))

    a2 = opt.sigmoid(np.matmul(X, np.transpose(Theta1)))
    a2 = np.column_stack((np.ones((m,)),a2))

    a3 = opt.sigmoid(np.matmul(a2, np.transpose(Theta2)))

    p = np.argmax(a3, axis=1) + 1
    return p
    




input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   

## =========== Part 1: Loading and Visualizing Data =============
mat = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex3/ex3/ex3data1.mat')
X =  mat['X']
y = mat['y']
y = np.squeeze(y)

rand_indices = np.random.permutation(X.shape[0])
sel = X[rand_indices[0:100], :]
# displayData(sel)

## ================ Part 2: Loading Pameters ================
weights = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex3/ex3/ex3weights.mat')
Theta1= weights['Theta1']
Theta2= weights['Theta2']
pred = predict(Theta1, Theta2, X)

comparison = pred == y
accuracy = np.mean(comparison.astype('float64'))
print("Train Accuracy: " + np.array2string(accuracy * 100))