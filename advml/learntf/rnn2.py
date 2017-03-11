#!/usr/bin/env python
# encoding: utf-8

"""
@description: 波形回归预测

@version: 1.0
@author: BaoQiang
@license: Apache Licence 
@contact: mailbaoqiang@gmail.com
@site: http://www.github.com/githubao
@software: PyCharm
@file: rnn2.py
@time: 2017/3/11 21:23
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LEARNING_RATES = 0.006
BATCH_START_TEST = 0


def get_batch():
    global BATCH_START, TIME_STEPS

    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))/10
    seq = np.sin(xs)
    res = np.cos(xs)

    BATCH_START += TIME_STEPS

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN():
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')

        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()
        self.compute_cost()

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATES).minimize(self.cost)

    def add_input_layer(self):
        # batch*n_step,in_size
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        Ws_in = self._weight_variables([self.input_size, self.cell_size])
        bs_in = self._bias_variables([self.cell_size, ])
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # (batch,n_step,cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False
        )

    def add_output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variables([self.cell_size, self.output_size], name='out_weights')
        bs_out = self._bias_variables([self.output_size, ], name='out_biases')

        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(self.pred, [-1], name="reshape_pred")],
                [tf.reshape(self.ys, [-1], name="reshape_target")],
                [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=self.msr_error,
                name="losses"
        )

        self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                tf.cast(self.batch_size, tf.float32),
                name='average_cost'
        )
        tf.summary.scalar('cost', self.cost)

    def msr_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variables(self, shape, name='my_weights'):
        initializer = tf.random_normal_initializer(mean=0, stddev=1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variables(self, shape, name='my_biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)


def cnn2():
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        plt.ion()
        plt.show()

        for i in range(200):
            seq, res, xs = get_batch()
            if i == 0:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state
                }

            _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict
            )

            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)

            if i % 20 == 0:
                print('cost: ', round(cost, 4))


def main():
    cnn2()


if __name__ == '__main__':
    main()

# cost:  17.3695
# cost:  0.9601
# cost:  0.2712
# cost:  0.0226