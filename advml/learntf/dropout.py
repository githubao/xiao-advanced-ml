#!/usr/bin/env python
# encoding: utf-8

"""
@description: 过拟合问题

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: dropout.py
@time: 2017/3/11 13:11
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from advml.learntf.func import add_layer

prediction = None

sess = tf.Session()

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


def digits_demo():
    global prediction

    digits = load_digits()
    X = digits.data
    y = digits.target
    y = LabelBinarizer().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    l1 = add_layer(xs, 64, 100, 'l1', activation_function=tf.nn.tanh)
    prediction = add_layer(l1, 100, 10, 'l2', activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('../log/train/', sess.graph)
    test_writer = tf.summary.FileWriter('../log/test/', sess.graph)

    for step in range(2000):
        # sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

        if step % 50 == 0:
            print(sess.run(cross_entropy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1}))
            # train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train})
            # test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
            #
            # train_writer.add_summary(train_result, step)
            # test_writer.add_summary(test_result, step)
            #
            # train_writer.flush()
            # test_writer.flush()

    train_writer.close()
    test_writer.close()


def main():
    digits_demo()


if __name__ == '__main__':
    main()
