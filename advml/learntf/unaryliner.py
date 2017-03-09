#!/usr/bin/env python
# encoding: utf-8

"""
@description: 一元线性插值

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: unaryliner.py
@time: 2017/3/8 22:25
"""

import tensorflow as tf
import numpy as np

def liner():
    # 真实值
    x_data = np.random.random(100)
    y_data = x_data * 0.1 + 0.3

    # 待学习的向量
    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))

    # 预测值
    y = Weights * x_data + biases

    # 损失
    loss = tf.reduce_mean(tf.square(y - y_data))

    # 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(0.5)

    # 训练的目标
    train = optimizer.minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 运行tf
    with tf.Session() as sess:
        sess.run(init)

        for step in range(201):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run(Weights), sess.run(biases))


def main():
    liner()


if __name__ == '__main__':
    main()
