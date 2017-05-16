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
    """
    def embedding_lookup(params, ids, partition_strategy="mod", name=None,validate_indices=True, max_norm=None):
    根据第二个参数的id，返回矩阵中对应的数据
    :return: 
    [[[0 1 0 0 0]
     [0 0 1 0 0]]
     [[0 0 0 0 1]
     [0 0 0 1 0]]]
    """

    inputs_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, inputs_ids)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(input_embedding, feed_dict={inputs_ids: [[1, 2], [4, 3]]})
        print(result)


def split_demo():
    """
    def split(value, num_or_size_splits, axis=0, num=None, name="split"):
    切分数据，返回多个tensor
    :return: 
    [(5, 4), (5, 15), (5, 11)]
    """

    inputs = tf.Variable(np.zeros([5, 30], dtype=np.float32))
    results = tf.split(inputs, [4, 15, 11], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print([item.shape for item in res])


def reduce_mean_demo():
    """
    axis=1,沿着第一维度平均，行相加
    :return: 
    [ 1.5  3.5]
    """
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    results = tf.reduce_mean(inputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def reduce_sum_demo():
    pass


def reshape_demo():
    """
    改变tensor的维度，-1表示自适应，空表示变为标量
    :return: 
    [[ 1.  2.]
    [ 3.  4.]
    [ 5.  6.]]
    
    [ 1.  2.  3.  4.  5.  6.]
    """
    inputs = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    results = tf.reshape(inputs, [3, -1])
    results2 = tf.reshape(inputs, [-1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(results))
        print(sess.run(results2))


def matmul_demo():
    tf.matmul(None, None)
    pass


def argmax_demo():
    """
        沿着指定的维度求最大值的索引，[0 2]
    :return: 
    """
    inputs = tf.Variable(np.array([[2, 1, 0], [3, 4, 5]], dtype=np.float32))
    results = tf.argmax(inputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def one_hot_demo():
    """
    沿着指定的indices构造出onthot向量出来，indice为-1表示全部给off_value,depth表示向量的长度
    [[[ 1.  0.  0.]
    [ 0.  0.  1.]]

    [[ 0.  1.  0.]
    [ 0.  0.  0.]]]
    :return: 
    """
    inputs = tf.Variable(np.array([[0, 2], [1, -1]], dtype=np.int32))
    results = tf.one_hot(inputs, depth=3, on_value=1.0, off_value=0.0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def cast_demo():
    """
    转化数据类型
    :return: 
    """
    tf.cast(tf.Variable(np.array([1, 2.2])), dtype=tf.int32)


def concat_demo():
    """
    连接起来
    :return: 
    [[ 1.  2.  5.  6.]
    [ 3.  4.  7.  8.]]
    """
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    inputs2 = tf.Variable(np.array([[5, 6], [7, 8]], dtype=np.float32))
    results = tf.concat([inputs, inputs2], axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def fill_demo():
    """
    对于给定的shape，全部填写相同的数据
    ones,zeros.ones_like(tensor),zeros_like 同理
    :return: 
    """
    tf.fill([2, 3], 2)


def random_normal_demo():
    """
    tf.random_normal(shape): 正态分布的随机数
    tf.truncated_normal(shape): 截断的正态分布的随机数 [mean-2*stddev,mean+2*stddev]
    tf.random_uniform(shape,minval,maxval):均匀分布随机数
    :return: 
    """
    pass


def get_variable_demo():
    """
    get_variable(name,shape,dtype,initializer): 如果有这个名字，返回向量，如果没有，根据给定的参数创建
    :return: 
    """
    pass


def expand_dim_demo():
    """
    扩展向量的维度，+1
    :return: 
    [[ 1.  2.  3.]]
    [[ 1.]
     [ 2.]
     [ 3.]]
    [[ 1.]
     [ 2.]
     [ 3.]]
    """
    inputs = tf.Variable(np.array([1, 2, 3], dtype=np.float32))
    results = tf.expand_dims(inputs, axis=0)
    results2 = tf.expand_dims(inputs, axis=1)
    results3 = tf.expand_dims(inputs, axis=-1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(results))
        print(sess.run(results2))
        print(sess.run(results3))


def sparse_to_dense():
    """
    稀疏矩阵 转化为 正常矩阵
    :return: 
    """
    pass


def sample_demo():
    """

    :return: 
    """
    inputs = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    results = tf.reduce_mean(inputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(results)
        print(res)


def main():
    # embedding_lookup_demo()
    # split_demo()
    # reduce_mean_demo()
    # reshape_demo()
    argmax_demo()
    # one_hot_demo()
    # concat_demo()
    # expand_dim_demo()


if __name__ == '__main__':
    main()
