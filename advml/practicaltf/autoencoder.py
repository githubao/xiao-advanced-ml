#!/usr/bin/env python
# encoding: utf-8

"""
@description: 自编码和多层感知机

@author: BaoQiang
@time: 2017/5/8 21:42
"""

'''
提取特征：

要识别一个汽车的图片，先看像素点，然后抽象出圆弧曲线等图形，然后抽象出轮胎，车轴，轮廓等
由底层的特征，不断向高层的特征，再向更高层的特征，一步一步地具体化

通过自编码的这种无监督学习先获取特征，获取初始权值的合理输入，然后再进行监督学习(比如rnn,cnn)计算

'''

import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
from advml.pth import *

mnist = input_data.read_data_sets('{}/MNIST_data'.format(FILE_PATH), one_hot=True)


# xavier 初始化器，使得数据梯度的计算不会消失或者弥散，而是使得初始化参数的取值正好合理
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder:
    def __init__(self, n_input, n_hidden, transfer_fn=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_fn
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initialize_weights()

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(
            tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
                   self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        all_weights = {
            'w1': tf.Variable(xavier_init(self.n_input, self.n_hidden)),
            'b1': tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32)),
            'w2': tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.n_input], dtype=tf.float32)),
        }
        return all_weights

    # 计算一个小的batch的值，并返回损失
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 最后计算的效果
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 隐含层的结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 重构层的结果
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # transform+generate
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 数据标准化
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def run():
    '''
    Epoch: 0001, cost=19564.939081
    Epoch: 0002, cost=13114.047061
    Epoch: 0019, cost=8081.920954
    Epoch: 0020, cost=7851.620235
    Total cost: 677097.0625    
    
    :return: 
    '''

    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_fn=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print('Epoch: {:04d}, cost={:.6f}'.format(epoch + 1, avg_cost))

    print('Total cost: {}'.format(autoencoder.calc_total_cost(X_test)))


def main():
    run()


if __name__ == '__main__':
    main()
