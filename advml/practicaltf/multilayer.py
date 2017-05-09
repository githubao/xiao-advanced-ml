#!/usr/bin/env python
# encoding: utf-8

"""
@description: 多层感知机

减轻过拟合的dropout
自适应学习速率的adagrad
解决梯度弥散的激活函数relu

抽象出隐含层的高阶特征：横线，竖线，圆圈等

FCN：全连接神经网络 full connected network
MLP：多层感知机 Multilayer Perceptron

@author: BaoQiang
@time: 2017/5/9 20:00
"""

from tensorflow.examples.tutorials.mnist import input_data
from advml.pth import *
import tensorflow as tf

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)


def train():
    '''
    [*,784] * [784,10] + [*,10] = [*,10]
    [2850]: 0.979200
    [2900]: 0.980600
    [2950]: 0.980100

    :return: 

    '''

    in_units = 784
    h1_units = 300

    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32)

    # 预测值
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

    # 真实值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 按行加起来求和，最后使得列的平均值最小
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # 训练
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    # 正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 3000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

            if i % 50 == 0:
                train_accuracy = sess.run(accuracy,
                                         feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1})
                print('[{:03d}]: {:03f}'.format(i, train_accuracy))

        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
        print('Final test accuracy: {:03g}'.format(test_accuracy))


def main():
    train()


if __name__ == '__main__':
    main()
