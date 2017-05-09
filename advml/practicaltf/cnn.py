#!/usr/bin/env python
# encoding: utf-8

"""
@description: 卷积层

局部连接
权值共享
池化层负采样

@author: BaoQiang
@time: 2017/5/9 20:20
"""

from tensorflow.examples.tutorials.mnist import input_data
from advml.pth import *
import tensorflow as tf

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)


def weight_variable(shape):
    # 截断的正态分布噪声，标准差0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '''
    W: [5,5,1,32],5*5 卷积核的大小，1：通道的总数(黑白图片是1，rgb图片是3)，32卷积核的数量
    stride: 采样时候的步长，是否跳过。1表示不会跳过。
    padding：same表示对于边上的接点，会外接一部分，保证输出与输入相同
    :param x: 
    :param W: 
    :return: 
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def run():
    '''
    step[0700] training accuracy: 0.980000
    step[0800] training accuracy: 0.920000
    step[0900] training accuracy: 0.980000
    :return: 
    '''

    # 定义变量
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 784 -> 28*28, 第四个参数1表示通道数量
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层
    # 28*28*32 -> 7*7*64
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    # 从三维变成1维的向量
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 误差
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for i in range(20000):
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
                print('step[{:04d}] training accuracy: {:0.6f}'.format(i, train_accuracy))

        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
        print('Final test accuracy: {:0.6f}'.format(test_accuracy))


def main():
    run()


if __name__ == '__main__':
    main()
