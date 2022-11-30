## Logistic regression example in "ML with TF MEAP" book
# Collected and modified by Alexander I. Iliev

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

learning_rate = 0.001
training_epochs = 200

# Defining the sigmoid function:
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# Create our data points on the x and y axis:
x1 = np.random.normal(5, 3, 100)
x2 = np.random.normal(-5, 3, 100)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))
plt.scatter(xs, ys)

# Create our parameters and placeholders for X and Y to feed them with the data above:
X = tf.placeholder(tf.float32, shape=(None,), name="x")
Y = tf.placeholder(tf.float32, shape=(None,), name="y")
w = tf.Variable([0., 0.], name="parameter", trainable=True)
y_model = tf.sigmoid(-(w[1] * X + w[0]))

# Calculate the cost and adaptation (learning):
cost = tf.reduce_mean(tf.log(y_model * Y + (1 - y_model) * (1 - Y)))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Run the model:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for epoch in range(training_epochs):
        err, _ = sess.run([cost, train_op], {X: xs, Y: ys}) # err = cost
        print(epoch, err)
        if abs(prev_err - err) < 0.0001: # adjust to see curve change with epochs
            break   # Check when the error is small enough to quit
        prev_err = err
    w_val = sess.run(w, {X: xs, Y: ys})

# Plot the resulting sigmoid:
all_xs = np.linspace(-10, 10, 100)
plt.plot(all_xs, sigmoid(all_xs * w_val[1] + w_val[0]), 'r') # calculate the sigmoid
plt.grid(), plt.title("Sigmoid function and the scatter points")
plt.pause(1)
