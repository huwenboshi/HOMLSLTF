import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

def main():

    print("here")

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.compat.v1.placeholder(tf.int64, shape=(None), name='y')

    """
    with tf.name_scope('dnn'):
        hidden1 = neuron_layer(X, n_hidden1, name='hidden1',
            activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2',
            activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name='outputs')
    """

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1',
            activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',
            activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
            logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')

    learning_rate = 0.01
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # load data
    #mnist = input_data.read_data_sets("/n/groups/price/huwenbo/EXPLORE/DL/HOMLSLTF/chp10/mnist")


if __name__ == '__main__':
    main()
