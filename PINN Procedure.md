# PINN Procedure

By Zeynep Sağlam, 2024

PINN stands for physics-based neural networks and is often used in scientific calculations, especially in solving differential equations. This procedure was developed as an alternative to traditional numerical analysis methods and is generally effective when data is sparse, noisy, or incomplete. In this notebook, the procedure required to create a Physics-informed neural network will be explained.

### Steps to follow:
1) Importing Required Libraries
2) Dataset Creation (Optional)
3) Definition of Neural Network Architecture
4) Creation of Physics-Informed Loss Functions
5) Main Program &  Plotting Results

## 1. Importing Required Libraries

After the necessary libraries are installed, version control of the required libraries is performed. This process ensures that the requirements are notified if the program to be implemented is shared. It is not a mandatory procedure.


```python
import tensorflow as tf
#or/both
import torch
import torch.nn as nn

import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns 
import codecs, json

tf.keras.backend.set_floatx('float64') # float64 as default

print("TensorFlow version: {}".format(tf.__version__))
print("Torch version: {}".format(torch.__version__))
```

    TensorFlow version: 2.14.0
    Torch version: 2.1.0+cu121


## 2. Dataset Creation (Optional)

When creating a PINN model, it is not always necessary to create a data set. Sometimes experimental/observational data and sometimes data produced in accordance with the differential equation are used to train the model. At this stage, it is decided how to meet the data set required to train the model and added to the program.


```python
# If the data is ready in a file, file reading operations are performed.
# This processing varies depending on the size and type of the data set, but basically;

inputs = []
outputs = []
file_name = 'dataset.npy' # 

with open('dataset.npy', 'rb') as f:
    inputs, outputs = np.load(f)
    
"""This particular usage is usually used to read the contents of a file (with the ".npy" extension)
created by the NumPy library and containing Numpy arrays. The "rb" mode indicates that the file contains 
binary data. These types of files are often used to store large amounts of digital data."""


# if data is to be created,


#from pyDOE import lhs   already did this on top

# Size of parameter space
num_samples = 100 # Number of samples
num_dimensions = 3 # Dimension of the parameter space

# Latin Hypercube Sampling 
lhs_samples = lhs(num_dimensions, samples=num_samples, criterion="maximin")

# Scaling values in the parameter space (for example, between 0 and 1)
param_min = 0
param_max = 1
scaled_lhs_samples = param_min + lhs_samples * (param_max - param_min)

# You can perform another operation using the obtained samples
# For example, calculating a function on combinations of parameters

# As an example, let's define a function (for example, a function that calculates the sum of the parameters)
def sample_function(parameters):
     return np.sum(parameters)

# Calculate the function value of each sample
function_values = np.apply_along_axis(sample_function, 1, scaled_lhs_samples)

# Print results
print("Data Set Created with Latin Hypercube Sampling:")
print("Parameters:", scaled_lhs_samples)
print("Function Values:", function_values)


# If this  transformed in a function

def trainingdata(N_u,N_f,N_v):
    
    #It is examined here in 2 dimensions, but the number of dimensions can be arranged as desired.
    
    leftedge_x = np.hstack((X[:,0][:,None], Y[:,0][:,None]))
    leftedge_u = usol[:,0][:,None]
    
    rightedge_x = np.hstack((X[:,-1][:,None], Y[:,-1][:,None]))
    rightedge_u = usol[:,-1][:,None]
    
    topedge_x = np.hstack((X[0,:][:,None], Y[0,:][:,None]))
    topedge_u = usol[0,:][:,None]
    
    bottomedge_x = np.hstack((X[-1,:][:,None], Y[-1,:][:,None]))
    bottomedge_u = usol[-1,:][:,None]
    
    all_X_u_train = np.vstack([leftedge_x, rightedge_x, bottomedge_x, topedge_x])
    all_u_train = np.vstack([leftedge_u, rightedge_u, bottomedge_u, topedge_u])  
     
    #choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u+N_v, replace=False) 
    
    X_u_train = all_X_u_train[idx[0:N_u], :] #choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx[0:N_u],:]      #choose corresponding u
    
    X_boundary_val = all_X_u_train[idx[N_u+1:N_u+N_v], :] #choose indices from  set 'idx' (x,t)
    u_boundary_val = all_u_train[idx[N_u+1:N_u+N_v],:]      #choose corresponding u
    
    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_f = lb + (ub-lb)*lhs(2,N_f+N_v) 
    X_f_train = np.vstack((X_f[0:N_f,:], X_u_train)) # append training points to collocation points 
    X_interior_val = X_f[N_f:N_f+N_v,:]
    
    return X_f_train, X_u_train, u_train, X_interior_val, X_boundary_val, u_boundary_val  
```

## 3. Definition of Neural Network Architecture

Artificial neural network architecture consists of an input layer, one or more hidden layers and an output layer. Each layer consists of neurons, and these neurons are connected to each other by weights and activation functions. The input layer represents the features in the data set, the hidden layers are used to understand and learn the data, and the output layer produces the results. Neurons multiply input information by weights, create the sum, and then pass it to an activation function to produce the output. This architecture has a wide range of applications for solving various problems and learning data sets.


```python
#this useful function uses tensorflow librariy

def generate_PINN(n,L):  

    #generate an array representing the network architecture
    layers = np.ones(1+L).astype(int)*n # 1 input layer + L hidden layers
    layers[0] = 2
    layers[-1] = 1

    #layers = [2, n1, n2, n3,........., n_L] #Network Configuration 

    PINN = tf.keras.Sequential()

    PINN.add(tf.keras.layers.InputLayer(input_shape=(layers[0],),name="input_layer", dtype = 'float64'))
    
    PINN.add(tf.keras.layers.Lambda(lambda X: 2*(X - lb)/(ub - lb) - 1)) 
    
    initializer = 'glorot_normal'     
    
    for l in range (len(layers)-2):

      # Xavier Initialization  
      PINN.add(tf.keras.layers.Dense(layers[l+1],kernel_initializer=initializer, bias_initializer='zeros',
                                  activation = tf.nn.tanh, name = "layer" + str(l+1), dtype = 'float64'))
        
    PINN.add(tf.keras.layers.Dense(layers[-1],kernel_initializer=initializer, bias_initializer='zeros',
                          activation = None, name = "output_layer" , dtype = 'float64'))    
        
    return PINN  

# If you use Pytorch;

def generate_PINN(n, L, lb, ub):
    # Generate an array representing the network architecture
    layers = [2] + [n] * L + [1]  # Network Configuration
    layers = [int(x) for x in layers]

    class PINN(nn.Module):
        def __init__(self):
            super(PINN, self).__init__()
            self.input_layer = nn.Linear(layers[0], layers[1])
            self.activation = nn.Tanh()

            # Xavier Initialization
            for l in range(1, len(layers) - 2):
                setattr(self, f"layer{l}", nn.Linear(layers[l], layers[l + 1]))

            self.output_layer = nn.Linear(layers[-2], layers[-1])

        def forward(self, X):
            X = self.input_layer(X)
            X = self.activation(X)

            for l in range(1, len(layers) - 2):
                X = getattr(self, f"layer{l}")(X)
                X = self.activation(X)

            X = self.output_layer(X)

            return X

    # Create the model
    PINN_model = PINN()

    # Instead of using a Lambda layer for normalizing input data, we directly perform normalization
    def normalize(x):
        return 2 * (x - lb) / (ub - lb) - 1

    return PINN_model, normalize

```

## 4. Creation of Physics-Informed Loss Functions

PINNs generally use two types of loss functions: PDE Loss and Boundary Loss. PDE Loss is used to solve differential equations and helps the model represent its behavior in a particular physical system. This loss function ensures that the model simulates the solution to the terms on the right-hand side of the differential equations as accurately as possible. Boundary Loss, on the other hand, ensures that the model behaves correctly under certain bounds or limiting conditions. Both types of loss are used in training the model, increasing its fit to observed data and allowing it to mimic physical equations more effectively. In this way, PINNs can not only fit the measured data but also accurately model the physical behavior inside the system.


```python
def pde_loss(model, input_data):
    # Here, a term expressing the differential equation is obtained from the output of the model
    # For example: differential_term = model(input_data) - some_derivative(input_data)

    differential_term = model(input_data) - some_derivative(input_data)

    # Loss function used to minimize the differential term
    return tf.reduce_mean(tf.square(differential_term))

def boundary_loss(model, boundary_data):

    # Here a term is obtained that expresses the difference between the output of the model and the correct values of the limits
    # For example: boundary_term = model(boundary_data) - some_boundary_values(boundary_data)

    boundary_term = model(boundary_data) - some_boundary_values(boundary_data)

    # Loss function used to minimize the difference with the true values of the limits
    return tf.reduce_mean(tf.square(boundary_term))

def total_loss(model, input_data, boundary_data):

    # Total loss is the sum of PDE Loss and Boundary Loss

    total_loss = pde_loss(model, input_data) + boundary_loss(model, boundary_data)

    return total_loss
```

## 5. Main Program &  Plotting Results

At this stage, after all the results are obtained, it is analyzed whether the program serves the desired purpose. Analysis of the results is basically done through three tables;

1. The results of the data with physical equations 
2. The results of the created model 
3. The absolute margin of error between these two results

If the results are not satisfactory, all procedures, equations and approaches should be reviewed.


```python
def solutionplot(u_pred,X_u_train,u_train):

    #Ground truth
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x_1, x_2, physics_sol, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Ground Truth $u(x_1,x_2)$', fontsize=15)


    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolor(x_1, x_2, prediction , cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Predicted $\hat u(x_1,x_2)$', fontsize=15)

    # Error
    
    
    plt.subplot(1, 3, 3)
    plt.pcolor(x_1, x_2, np.abs(physics_sol - prediction), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(r'Absolute error $|u(x_1,x_2)- \hat u(x_1,x_2)|$', fontsize=15)
    plt.tight_layout()

    plt.savefig('Results.png', dpi = 500, bbox_inches='tight')
    

""" MAIN PROGRAM """

N_u = 400 #Total number of data points for the function
N_f = 10000 #Total number of collocation points 

# Training data
X_f_train_np_array, X_u_train_np_array, u_train_np_array = trainingdata(N_u,N_f)

'Convert to tensor and send to GPU'
X_f_train = torch.from_numpy(X_f_train_np_array).float()
X_u_train = torch.from_numpy(X_u_train_np_array).float()
u_train = torch.from_numpy(u_train_np_array).float()
X_u_test_tensor = torch.from_numpy(X_u_test).float()
u = torch.from_numpy(u_true).float()
f_hat = torch.zeros(X_f_train.shape[0],1)

PINN=generate_PINN(50,3)   #3 hidden layers, 50 neurons

'Neural Network Summary'

print(PINN)

params = list(PINN.parameters())

'''Optimization'''

'L-BFGS Optimizer'

'Adam Optimizer'

optimizer = optim.Adam(PINN.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 1000

start_time = time.time()

for i in range(max_iter):

    loss = PINN.loss(X_u_train, u_train, X_f_train)
           
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    
    loss.backward() #backprop

    optimizer.step()
    
    if i % (max_iter/10) == 0:

        error_vec, _ = PINN.test()

        print(loss,error_vec)
    
    
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))


''' Model Accuracy ''' 
error_vec, prediction = PINN.test()

print('Test Error: %.5f'  % (error_vec))


''' Solution Plot '''
solutionplot(prediction,X_u_train,u_train)
```
