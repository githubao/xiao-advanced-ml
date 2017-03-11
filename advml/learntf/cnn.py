#!/usr/bin/env python
# encoding: utf-8

"""
@description: 卷积神经网络

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: cnn.py
@time: 2017/3/11 15:10
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from advml.pth import *

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)

prediction = None
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()


def cnn():
    global prediction

    # 最后一个1，表示图片是黑白的
    # -1表示不管有多少个，每个的大小是 28*28*1
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    # 厚度从1变成了32
    # patch：5*5 insize:1 outsize:32
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 第一个卷积层
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # pooling之后
    h_pool1 = max_pool_2x2(h_conv1)

    # 再来一层，第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    # func1 layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # 把pooling之后的结果变平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # func2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print('step{:03d}: {:0.5f}'.format(i, compute_accuracy(mnist.test.images, mnist.test.labels)))

    sess.close()


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, W):
    # strides = [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def compute_accuracy(v_xs, v_ys):
    global prediction

    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def main():
    cnn()


if __name__ == '__main__':
    main()

# step0: 0.08230
# step50: 0.78870
# step100: 0.87870
# step150: 0.90520
# step200: 0.92300
# step250: 0.93330
# step300: 0.93920
# step350: 0.94280
# step400: 0.94760
# step450: 0.95200
# step500: 0.95340
# step550: 0.95650