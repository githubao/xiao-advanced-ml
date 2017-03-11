#!/usr/bin/env python
# encoding: utf-8

"""
@description: tenforflow 的 分类实现

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: classification.py
@time: 2017/3/10 12:25
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from advml.learntf.func import add_layer
from advml.pth import *

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)

prediction = None
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
sess = tf.Session()


def classify():
    global prediction

    prediction = add_layer(xs, 784, 10, 'l1',activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print('step {}: {}'.format(i, compute_accuracy(mnist.test.images, mnist.test.labels)))

    sess.close()


def compute_accuracy(v_xs, v_ys):
    global prediction

    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def main():
    classify()


if __name__ == '__main__':
    main()
