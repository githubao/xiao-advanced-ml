#!/usr/bin/env python
# encoding: utf-8

"""
@description: tensorflow 实现的Word2vec

@author: BaoQiang
@time: 2017/5/11 16:55
"""

import collections
import math
import os
import random
import zipfile
import urllib.request
import numpy as np
import tensorflow as tf
from advml.pth import FILE_PATH
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

filename = '{}/word2vec/text8.zip'.format(FILE_PATH)

url = 'http://mattmahoney.net/dc/'

vocabulary_size = 50000


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify {}. Can you get to it with a browser?')
    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


maybe_download(filename, 31344016)
words = read_data(filename)
print('Data size: ', len(words))
data, count, dictionary, reversed_dictionary = build_dataset(words)

del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


'''
test
'''
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0], reversed_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64


def train():
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size
        ))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    # num_steps = 100001
    num_steps = 50001
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step {} : {:0.6f}'.format(step, average_loss))
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reversed_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to {} :'.format(valid_word)
                    for k in range(top_k):
                        close_word = reversed_dictionary[nearest[k]]
                        log_str = '{} {}, '.format(log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()
        return final_embeddings


def plot_with_labels(low_dim_embs, labels, filename='{}/tsne.png'.format(FILE_PATH)):
    assert low_dim_embs.shape[0] >= len(labels), 'More label than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


def save_png(final_embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 100
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)


def main():
    final_embeddings = train()
    save_png(final_embeddings)


if __name__ == '__main__':
    main()
