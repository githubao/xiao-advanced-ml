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
import matplotlib.pyplot as plt


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

    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # layer1 隐藏层 输入层维度*隐藏层维度
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

    # 输出层
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

    # 损失
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)

    # 训练
    # 不同的学习优化器，对应于不同的学习效率
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(learning_rate=0.1).minimize(loss)
        # train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        # train_step = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 绘图
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../logs/', sess.graph)
        writer.add_graph(sess.graph)
        writer.close()

        sess.run(init)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # 真实的数据
        ax.scatter(x_data, y_data)
        # 不让函数暂停挂起
        plt.ion()
        plt.show()

        for step in range(1000):
            result = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

            if step % 50 == 0:
                # print('step{}: {}'.format(step, result))
                # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

                try:
                    ax.lines.remove(lines[0])
                except Exception as e:
                    pass

                prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

                plt.pause(0.1)

                merged_result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
                writer.add_summary(merged_result, step)

                writer.close()


# 优化器
def optimize():
    pass


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer{}'.format(n_layer)
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


def tf_board():
    build_nn()
    # tensorboard --logdir='logs/'


def main():
    # mat_mul()
    # variable()
    # place_hold()
    # active_func()
    # build_nn()
    tf_board()


if __name__ == '__main__':
    main()
