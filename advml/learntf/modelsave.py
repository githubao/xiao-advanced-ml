#!/usr/bin/env python
# encoding: utf-8

"""
@description: 保存模型

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: modelsave.py
@time: 2017/3/11 16:17
"""

import tensorflow as tf
from advml.pth import *
import numpy as np

model_path = '{}/my_net/tf_nn_net.ckpt'.format(FILE_PATH)


def save():
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, model_path)

        print(save_path)


def load():
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32,name='weights')
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32,name='biases')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        print(sess.run(W))
        print(sess.run(b))


def main():
    # save()
    load()


if __name__ == '__main__':
    main()
