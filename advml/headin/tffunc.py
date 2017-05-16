#!/usr/bin/env python
# encoding: utf-8

"""
@description: numpy 和 tensorflow的一些公式

@author: BaoQiang
@time: 2017/5/15 16:47
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.flags


def embedding_lookup_demo():
    '''
    根据第二个参数的id，返回矩阵中对应的数据
    :return: 
    [[[0 1 0 0 0]
     [0 0 1 0 0]]
     [[0 0 0 0 1]
     [0 0 0 1 0]]]
    '''

    inputs_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, inputs_ids)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(input_embedding, feed_dict={inputs_ids: [[1, 2], [4, 3]]})
        print(result)


def main():
    embedding_lookup_demo()


if __name__ == '__main__':
    main()
