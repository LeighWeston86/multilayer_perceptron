import tensorflow as tf
from data.data_utils import get_data
import numpy as np
import math

def initialize_weights(layer_sizes, n_input, n_classes):
    '''
    Initialize the weights and biases
    :param layer_sizes: list; size of each hidden layer
    :return: weights, biases
    '''

    #Weight and bias for the first layer
    weights = {
        'w1' : tf.Variable(tf.random_normal([layer_sizes[0], n_input]))
    }
    biases = {
        'b1' : tf.Variable(tf.random_normal([layer_sizes[0], 1]))
    }

    #Weights and bias for other hidden layers
    for idx in range(1, len(layer_sizes)):
        weights['w{}'.format(idx+1)] = tf.Variable(tf.random_normal([layer_sizes[idx], layer_sizes[idx-1]]))
        biases['b{}'.format(idx+1)]  = tf.Variable(tf.random_normal([layer_sizes[idx], 1]))

    #Weights for the output layer
    weights['out'] = tf.Variable(tf.random_normal([n_classes, layer_sizes[-1]]))
    biases['out']  = tf.Variable(tf.random_normal([1, n_classes]))

    return weights, biases


def forward_prop(X, weights, biases, dropout):
    '''
    Implement forward propagation.
    :return: output activations
    '''

    #First layer
    Z1 = tf.add(tf.matmul(weights['w1'], X), biases['b1'])
    A1 = tf.nn.relu(Z1)
    A1 = tf.layers.dropout(A1, dropout)

    #Second layer
    Z2 = tf.add(tf.matmul(weights['w2'], A1), biases['b2'])
    A2 = tf.nn.relu(Z2)
    A2 = tf.layers.dropout(A2)

    #Third layer
    Z3 = tf.add(tf.matmul(weights['w3'], A2), biases['b3'])
    A3 = tf.nn.relu(Z3)
    A3 = tf.layers.dropout(A3)

    #Output layer
    out = tf.add(tf.matmul(weights['out'], A3), biases['out'])

    return out

def calculate_cost(out, y):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y)
    mean_cost = tf.reduce_mean(cost)
    return mean_cost

def get_minibatches(X, y, batch_size):
    num_batches = math.ceil(X.shape[1]/batch_size)
    X_batches = np.array_split(X, num_batches, axis = 1)
    y_batches = np.array_split(y, num_batches, axis=1)
    return [(X_batch, y_batch) for X_batch, y_batch in zip(X_batches, y_batches)]

def fit_model(X_train,
              y_train,
              layer_sizes,
              dropout = 0.5,
              learning_rate = 0.001,
              epochs = 100,
              minibatch_size = 128):

    #Create placeholders
    X = tf.placeholder(tf.float32, shape = [X_train.shape[0], None])
    y = tf.placeholder(tf.float32, shape = [1, None])

    #Initialize the weights
    n_input = X_train.shape[0]
    n_classes = 1
    weights, biases = initialize_weights(layer_sizes, n_input, n_classes)

    #Forward prop, cost and optimizer
    out = forward_prop(X, weights, biases, dropout)
    cost = calculate_cost(out, y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Start the tf session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        #Initialize
        sess.run(init)

        #Loop over epochs
        seed = 0
        for epoch in range(epochs):

            #Define a set of minibatches
            num_minibatches = int(X_train.shape[1]/minibatch_size)
            minibatches = get_minibatches(X_train, y_train, minibatch_size)
            epoch_cost = 0
            for minibatch in minibatches:
                batch_X, batch_y = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: batch_X, y: batch_y})
                epoch_cost += minibatch_cost/num_minibatches

            #Print cost after each epoch
            if epoch % 10 == 0:
                print(epoch_cost)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    layer_sizes = [50, 100, 50]
    dropout = 0.5
    learning_rate = 0.0000001
    fit_model(X_train.T, y_train.reshape(1, -1), layer_sizes, dropout, learning_rate, epochs = 10000)
    ##To do include metrics















