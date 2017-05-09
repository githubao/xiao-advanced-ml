#!/usr/bin/env python
# encoding: utf-8

"""
@description: 再写一遍mnist的源代码

@author: BaoQiang
@time: 2017/5/8 20:56
"""

from tensorflow.examples.tutorials.mnist import input_data
from advml.pth import *
import tensorflow as tf

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)


def train():
    '''
    [*,784] * [784,10] + [*,10] = [*,10]
    :return: 
    [850]: 0.918700
    [900]: 0.915800
    [950]: 0.919800
    '''

    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 预测值
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 真实值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 按行加起来求和，最后使得列的平均值最小
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # 训练
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 50 == 0:
                test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print('[{:03d}]: {:03f}'.format(i, test_accuracy))


def data_info():
    '''
    (55000, 784) (55000, 10)
    (10000, 784) (10000, 10)
    (5000, 784) (5000, 10)
    :return: 
    '''
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)


def main():
    # data_info()
    train()


if __name__ == '__main__':
    main()
