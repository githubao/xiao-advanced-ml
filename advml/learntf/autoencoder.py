#!/usr/bin/env python
# encoding: utf-8

"""
@description:encoder and decoder

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: sec02_autoencoder.py
@time: 2017/3/13 19:18
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from advml.pth import *

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)

# learning_rate = 0.01
# training_epochs = 5
learning_rate = 0.001
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

n_input = 784

X = tf.placeholder('float', [None, n_input])

n_hidden_1 = 256
# n_hidden_2 = 128
n_hidden_2 = 2
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    return layer_2


def en_decode():
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op
    y_true = X

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

            if epoch % display_step == 0:
                print('Epoch: {:04d}, cost: {:.9f}'.format(epoch + 1, c))

        print('Finished')

        encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[1], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[1], (28, 28)))
        plt.show()


def main():
    en_decode()


if __name__ == '__main__':
    main()

    # Epoch: 0001, cost: 0.092974178
    # Epoch: 0002, cost: 0.080574177
    # Epoch: 0003, cost: 0.072209008
    # Epoch: 0004, cost: 0.067358822
    # Epoch: 0005, cost: 0.063710891
    # Finished
