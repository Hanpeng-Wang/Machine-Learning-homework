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

def debugInitializeWeights(fan_out, fan_in):
    tmp = np.sin(np.arange(1, fan_out * (fan_in + 1) + 1))
    W = tmp.reshape((fan_out, 1 + fan_in)) /10.0
    return W

def checkNNGradients(lamda=None):
    if lamda is None:
        lamda = 0
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = np.arange(1, m+1) % num_labels + 1

    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    cost, grad = opt.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)

    pertub = np.zeros((len(nn_params), ))
    numgrad = np.zeros((len(nn_params), ))
    e = 1e-4
    for i in range(len(nn_params)):
        pertub[i] = e
        loss1, _ = opt.nnCostFunction(nn_params-pertub, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
        loss2, _ = opt.nnCostFunction(nn_params+pertub, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
        numgrad[i] = (loss2 - loss1)/(2*e)
        pertub[i] = 0

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print("If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).\n", "Relative Difference: ", diff)

    plt.plot(np.arange(len(nn_params)), numgrad, 'g*', label="numgrad")
    plt.plot(np.arange(len(nn_params)), grad, 'r', label="grad")
    plt.legend()
    plt.show()

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.column_stack((np.ones((m,)),X))
    z2 = np.matmul(a1, Theta1.transpose())
    a2 = opt.sigmoid(z2)

    a2 = np.column_stack((np.ones((a2.shape[0],)),a2))
    z3 = np.matmul(a2, Theta2.transpose())
    a3 = opt.sigmoid(z3)

    p = np.argmax(a3, axis=1) + 1
    return p




input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   


## =========== Loading and Visualizing Data =============
mat = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex4/ex4/ex4data1.mat')
print(type(mat))
X =  mat['X']
y = mat['y']
y = np.squeeze(y)
# rand_indices = np.random.permutation(X.shape[0])
# sel = X[rand_indices[0:100], :]


# displayData(sel)


#  %%================ Part 2: Loading Parameters ================
weights = io.loadmat('/Users/hanpengwang/Documents/Machine-Learning-homework/machine-learning-ex4/ex4/ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

# %% ================== Part 3: Compute Cost (Feedforward) ================
lamda = 0
J, _ = opt.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
print("Cost at parameters (loaded from ex4weights): "+ np.array2string(J))
print("Expected cost: 0.287629")


lamda = 1
J, _ = opt.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
print("Cost at parameters (loaded from ex4weights): "+ np.array2string(J))
print("Expected cost: 0.383770")

# %% ================ Part 5: Sigmoid Gradient  ================
g = opt.sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: " + np.array2string(g))

# %% ================ Part 6: Initializing Pameters ================

initial_Theta1 = opt.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = opt.randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))


checkNNGradients()

# %% =============== Part 8: Implement Regularization ===============

lamda = 3
checkNNGradients(lamda)

debug_J, _  = opt.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
print("Cost at (fixed) debugging parameters (w/ lambda = %f): %f"%(lamda, debug_J), "(for lambda = 3, this value should be about 0.576051)")

# %% =================== Part 8: Training NN ===================
lamda = 1
myoptions={"disp": True,
            "maxiter": 100}
res = minimize(opt.nnCostFunction, initial_nn_params, method='BFGS', jac=True, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamda), 
                options=myoptions)

Theta1 = np.reshape(res.x[0:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, (input_layer_size + 1)))
Theta2 = np.reshape(res.x[hidden_layer_size * (input_layer_size + 1):],(num_labels, (hidden_layer_size + 1)))

pred = predict(Theta1, Theta2, X)

comparison = pred == y
accuracy = np.mean(comparison.astype('float64'))
print("Train Accuracy: " + np.array2string(accuracy * 100))

