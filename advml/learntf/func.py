#!/usr/bin/env python
# encoding: utf-8

"""
@description: 基础功能函数

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: func.py
@time: 2017/3/11 14:45
"""

import tensorflow as tf

def add_layer(inputs, in_size, out_size,layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name+'outputs',outputs)
    return outputs

