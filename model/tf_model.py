import tensorflow as tf
from data.data_utils import get_data
from data.mnist import get_mnist
import numpy as np
import math

def initialize_weights():
    '''Model has 3 hidden layers of size 100, 150, 50;
    Input layer has size 784, output layer has size 10'''
    weights = {}

    #1st hidden layer
    weights['w_1'] = tf.Variable(tf.random_normal(shape = [784, 100]))
    weights['b_1'] = tf.Variable(tf.random_normal(shape = [100]))

    #2nd hidden layer
    weights['w_2'] = tf.Variable(tf.random_normal(shape = [100, 150]))
    weights['b_2'] = tf.Variable(tf.random_normal(shape = [150]))

    #3rd hidden layer
    weights['w_3'] = tf.Variable(tf.random_normal(shape = [150, 50]))
    weights['b_3'] = tf.Variable(tf.random_normal(shape = [50]))

    #output layer
    weights['w_out'] = tf.Variable(tf.random_normal(shape = [50, 10]))
    weights['b_out'] = tf.Variable(tf.random_normal(shape=[10]))

    return weights

def neural_network(X, weights):

    #1st hidden layer
    Z1 = tf.add(tf.matmul(X, weights['w_1']), weights['b_1'])
    A1 = tf.nn.relu(Z1)

    #2nd hidden layer
    Z2 = tf.add(tf.matmul(A1, weights['w_2']), weights['b_2'])
    A2 = tf.nn.relu(Z2)

    #3rd hidden layer
    Z3 = tf.add(tf.matmul(A2, weights['w_3']), weights['b_3'])
    A3 = tf.nn.relu(Z3)

    #output layer
    out = tf.add(tf.matmul(A3, weights['w_out']), weights['b_out'])

    return out

def get_minibatches(X, y, batch_size):
    num_batches = math.ceil(X.shape[1]/batch_size)
    X_batches = np.array_split(X, num_batches, axis = 0)
    y_batches = np.array_split(y, num_batches, axis=0)
    return [(X_batch, y_batch) for X_batch, y_batch in zip(X_batches, y_batches)]

def fit_model(X_train, X_test, y_train, y_test, learning_rate = 0.001, batch_size = 32, epochs = 5):

    #Shape
    num_features = X_train.shape[1]
    num_classes  = y_train.shape[1]

    #Create the placeholders
    X = tf.placeholder(dtype = tf.float32, shape = [None, num_features])
    y = tf.placeholder(dtype = tf.float32, shape = [None, num_classes])

    #Initialize the weights
    weights = initialize_weights()

    #Define the logits
    out = neural_network(X, weights)

    #Define the cost and optimizer)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    #initialize
    init = tf.global_variables_initializer()

    #Run the session
    with tf.Session() as sess:

        #Initialize
        sess.run(init)

        for epoch in range(epochs):
            #Get the minibatches
            minibatches = get_minibatches(X_train, y_train, batch_size)
            num_batches = len(minibatches)
            epoch_cost = 0
            for minibatch in minibatches:
                X_batch, y_batch = minibatch
                _, batch_cost = sess.run([optimizer, cost], feed_dict= {X:X_batch, y:y_batch})
                epoch_cost += batch_cost/num_batches
            print(epoch, epoch_cost)

        # Test model
        pred = tf.nn.softmax(out)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist()
    fit_model(X_train, X_test, y_train, y_test)
    #layer_sizes = [50, 100, 50]
    #dropout = 0.1
    #learning_rate = 0.001
    #fit_model(X_train.T, y_train.reshape(1, -1), X_test.T, y_test.reshape(1, -1), layer_sizes, dropout, learning_rate, epochs = 1000)