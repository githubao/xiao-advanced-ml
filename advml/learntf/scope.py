#!/usr/bin/env python
# encoding: utf-8

"""
@description: name_scope 和 variable_scope

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: scope.py
@time: 2017/3/13 19:46
"""

import tensorflow as tf

tf.set_random_seed(1)


# 对get_variable方式，没有命名空间
# 对同名的变量，第二个机器以后，会重新命名一个值
# var1:0
# [ 1.]
# a_name_scope/var2:0
# [ 2.]
# a_name_scope/var2_1:0
# [ 2.0999999]
def name_scope_demo():
    with tf.name_scope('a_name_scope'):
        initializer = tf.constant_initializer(value=1)
        var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
        var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
        var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(var1.name)
        print(sess.run(var1))
        print(var2.name)
        print(sess.run(var2))
        print(var21.name)
        print(sess.run(var21))


# 命名空间对所有变量有效
# get_variable 变量被重复使用
# a_variable_scope/var1:0
# [ 1.]
# a_variable_scope/var2:0
# [ 2.]
# a_variable_scope/var2_1:0
# [ 2.0999999]
def variable_scope_demo():
    with tf.variable_scope('a_variable_scope') as scope:
        initializer = tf.constant_initializer(value=1)
        var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
        # 重复使用变量
        scope.reuse_variables()
        var11 = tf.get_variable(name='var1')
        var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
        var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(var1.name)
        print(sess.run(var1))
        print(var11.name)
        print(sess.run(var11))
        print(var2.name)
        print(sess.run(var2))
        print(var21.name)
        print(sess.run(var21))


class TrainConfig:
    batch_size = 20
    time_steps = 10
    # ...


class TestConfig(TrainConfig):
    batch_size = 2
    time_steps = 1


class RNN():
    def __init__(self, config):
        self._batch_size = config.batch_size
        self._time_steps = config.time_steps

        initializer = tf.constant_initializer(value=1)
        self.weight = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)

        self.print_weight()

    def print_weight(self):
        print(self.weight.name)


def reuse():
    train_config = TrainConfig()
    test_config = TestConfig()

    with tf.variable_scope('rnn') as scope:
        sess = tf.Session()
        train_rnn = RNN(train_config)
        scope.reuse_variables()
        # 所有的参数都共享之前的参数
        test_rnn = RNN(test_config)


def main():
    # name_scope_demo()
    # variable_scope_demo()
    reuse()


if __name__ == '__main__':
    main()
