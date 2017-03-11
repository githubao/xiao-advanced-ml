#!/usr/bin/env python
# encoding: utf-8

"""
@description: 循环递归神经网络

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: rnn.py
@time: 2017/3/11 18:10
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from advml.pth import *

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)

learning_rate = 0.001
train_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10


def rnn():
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ])),
    }

    pred = rnn_func(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < train_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
            sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})

            if step % display_step == 0:
                print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))


def rnn_func(X, weights, biases):
    # input
    # X(128,28,28) ++> (128*28,28)
    X = tf.reshape(X, [-1, n_input])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X(128*28,128) => (128,28,128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # lstm_cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # state(主线剧情，分线剧情)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # output
    result = tf.matmul(states[1], weights['out']) + biases['out']
    # outputs = tf._unpack(tf.transpose(outputs, [1, 0, 2]))
    # result = tf.matmul(outputs[-1],weights['out']) + biases['out']
    return result


def main():
    rnn()


if __name__ == '__main__':
    main()

# 0.945313
# 0.9375
# 0.851563
# 0.859375
# 0.960938
# 0.945313
# 0.867188
# 0.90625
# 0.90625
# 0.914063
# 0.96875