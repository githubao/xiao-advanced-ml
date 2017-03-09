#!/usr/bin/env python
# encoding: utf-8

"""
@description: 教程的一些简单的demo

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: tutorial.py
@time: 2017/3/9 19:59
"""

import tensorflow as tf
import numpy as np


def mat_mul():
    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2],
                           [2]])

    product = tf.matmul(matrix1, matrix2)
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)


def variable():
    state = tf.Variable(0, name='counter')
    # print(state.name)

    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for _ in range(3):
            sess.run(update)
            print(sess.run(state))


def place_hold():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7], input2: [2]}))


# 使一些值先激活
def active_func():
    pass


def build_nn():
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # layer1 隐藏层 输入层维度*隐藏层维度
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    # 输出层
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # 损失
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    # 训练
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(1000):
            result = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

            if step % 50 == 0:
                # print('step{}: {}'.format(step, result))
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


def main():
    # mat_mul()
    # variable()
    # place_hold()
    # active_func()
    build_nn()


if __name__ == '__main__':
    main()
