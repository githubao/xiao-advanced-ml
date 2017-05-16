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
    def embedding_lookup(params, ids, partition_strategy="mod", name=None,validate_indices=True, max_norm=None):
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


def split_demo():
    '''
    def split(value, num_or_size_splits, axis=0, num=None, name="split"):
    切分数据，返回多个tensor
    :return: 
    [(5, 4), (5, 15), (5, 11)]
    '''

    inputs = tf.Variable(np.zeros([5, 30], dtype=np.float32))
    results = tf.split(inputs, [4, 15, 11], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print([item.shape for item in res])


def reduce_mean_demo():
    '''
    axis=1,沿着第一维度平均，行相加
    :return: 
    [ 1.5  3.5]
    '''
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    results = tf.reduce_mean(inputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def reduce_sum_demo():
    pass


def reshape_demo():
    '''
    改变tensor的维度，-1表示自适应，空表示变为标量
    :return: 
    [[ 1.  2.]
    [ 3.  4.]
    [ 5.  6.]]
    
    [ 1.  2.  3.  4.  5.  6.]
    '''
    inputs = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    results = tf.reshape(inputs, [3, -1])
    results2 = tf.reshape(inputs, [-1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(results))
        print(sess.run(results2))


def matmul_demo():
    pass


def argmax_demo():
    '''
        沿着指定的维度求最大值的索引，[0 1]
    :return: 
    '''
    inputs = tf.Variable(np.array([[2, 1], [3, 4]], dtype=np.float32))
    results = tf.argmax(inputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def one_hot_demo():
    '''
    沿着指定的indices构造出onthot向量出来，indice为-1表示全部给off_value,depth表示向量的长度
    [[[ 1.  0.  0.]
    [ 0.  0.  1.]]

    [[ 0.  1.  0.]
    [ 0.  0.  0.]]]
    :return: 
    '''
    inputs = tf.Variable(np.array([[0, 2], [1, -1]], dtype=np.int32))
    results = tf.one_hot(inputs, depth=3, on_value=1.0, off_value=0.0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def cast_demo():
    '''
    转化数据类型
    :return: 
    '''
    tf.cast(tf.Variable(np.array([1, 2.2])), dtype=tf.int32)


def concat_demo():
    '''
    连接起来
    :return: 
    [[ 1.  2.  5.  6.]
    [ 3.  4.  7.  8.]]
    '''
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    inputs2 = tf.Variable(np.array([[5, 6], [7, 8]], dtype=np.float32))
    results = tf.concat([inputs, inputs2], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def sample_demo():
    '''

    :return: 
    '''
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    results = tf.reduce_mean([inputs, inputs2], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def main():
    # embedding_lookup_demo()
    # split_demo()
    # reduce_mean_demo()
    # reshape_demo()
    # argmax_demo()
    # one_hot_demo()
    concat_demo()


if __name__ == '__main__':
    main()
