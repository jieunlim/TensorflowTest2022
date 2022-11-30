# coding=utf-8
"""
This is an atempt to exploit the functionality of a NN
    Collected and modified by: Alexander Iliev
    Main code:
        Source 1: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
    Fixing the 'check_array' to 'check_arrays' problem at:
        Source 2: http://stackoverflow.com/questions/29596237/import-check-arrays-from-sklearn
    Function def 'plot_decision_boundary' explained at:
        Source 3: http://stackoverflow.com/questions/34829807/understand-how-this-lambda-function-works
"""
## Generate a dataset and plot it:
from numpy import random as rnd
from matplotlib import pyplot as plt
from sklearn.datasets import samples_generator as sks

rnd.seed(0)

# Alex: Run for 2 clusters:
X, y = sks.make_moons(200, noise=0.20)
# Alex: Run for 3 clusters for log. reg. only: 
# X, y = sks.make_blobs(200)

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.pause(5)

## Train the logistic regeression classifier:
# See source 2 for the fix #
import numpy as np
from sklearn.linear_model import logistic as lgc

# See source 3 for func.def & explanation:
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
clf = lgc.LogisticRegressionCV()
clf.fit(X, y)
 
# Plot the decision boundary
plt.figure(2)
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.pause(1)

## We build a 3-layer neural network with: 1 input layer, 1 hidden layer, and 1 output layer:

# 1.Lets create the parameters for gradient descent:
num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (We pick these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# 2. Let’s implement the loss function and use it to evaluate how well our model is doing.
# Helper function to evaluate the total loss on the dataset:
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
    
# 3. We also implement a helper function to calculate the output of the network. 
# It does forward propagation and returns the class with the highest probability:
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# 4. We finally create the function to train our Neural Network. It implements 
# batch gradient descent using the backpropagation derivates we found above:
 # This function learns parameters for the neural network and returns the model.
 # - nn_hdim: Number of nodes in the hidden layer
 # - num_passes: Number of passes through the training data for gradient descent
 # - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # # Optionally print the loss.
        # # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
     
    return model
    
## 5. Execution step for the implemented NN above using different number of layers:

# 5.1.1. Build a model with a 1-dimensional hidden layer:
model = build_model(1, print_loss=True)
# Plot the decision boundary
plt.figure(3)
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 1")
plt.pause(1)

# 5.1.2. Build a model with a 3-dimensional hidden layer:
model = build_model(3, print_loss=True)
# Plot the decision boundary
plt.figure(4)
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.pause(1)

# 5.1.3. Build a model with a 6-dimensional hidden layer:
model = build_model(6, print_loss=True)
# Plot the decision boundary
plt.figure(5)
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 6")
plt.pause(1)
# 
# # 5.2. Let’s get a sense of how varying the hidden layer size affects the result:
# plt.figure(6,figsize=(8, 16))
# hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
# for i, nn_hdim in enumerate(hidden_layer_dimensions):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer size %d' % nn_hdim)
#     model = build_model(nn_hdim)
#     plot_decision_boundary(lambda x: predict(model, x))
# plt.pause(2)

# Note: We can see that a hidden layer of low dimensionality nicely captures the general trend of our data. Higher dimensionalities are prone to overfitting. They are “memorizing” the data as opposed to fitting the general shape. If we were to evaluate our model on a separate test set (and you should!) the model with a smaller hidden layer size would likely perform better due to better generalization. We could counteract overfitting with stronger regularization, but picking the a correct size for hidden layer is a much more “economical” solution.