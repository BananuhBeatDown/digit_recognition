#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:47:41 2017

@author: matthew_green
"""

from __future__ import print_function
from pickle_work_around import pickle_load
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import tensorflow as tf
import random

pickle_file = 'metas/MNIST_multi_32.pickle'

load = pickle_load(pickle_file)
train_dataset = load['train_dataset']
train_labels = load['train_labels']
valid_dataset = load['valid_dataset']
valid_labels = load['valid_labels']
test_dataset = load['test_dataset']
test_labels = load['test_labels']
del load  # hint to help gc free up memory

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# %%
    
# DIGIT AND BOUNDING BOX DISPLAY TEST:

def displaySequence_test(n):
    fig,ax=plt.subplots(1)
    plt.imshow(train_dataset[n].reshape(32, 32), cmap=plt.cm.Greys)
    plt.show
    print ('Label : {}'.format(train_labels[n], cmap=plt.cm.Greys), n)
    print(n)
    
    
# display random sample to check if data is ok after creating sequences
displaySequence_test(random.randint(0, train_dataset.shape[0] - 1))

# %%

# INPUT LAYER:

def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.int32, shape=(None, n_classes), name='y')

def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob') 


tf.reset_default_graph()

# %%

# CONVOLUTION AND POOLING LAYER

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weight = tf.Variable(
                 tf.truncated_normal(
                     shape=[conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                     mean=0.0,
                     stddev=0.1))
    bias = tf.Variable(tf.zeros(shape=conv_num_outputs))
    
    conv = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    hidden = tf.nn.relu(conv + bias)
    pool = tf.nn.max_pool(hidden,
                         ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                         strides=[1, pool_strides[0], pool_strides[1], 1],
                         padding='SAME')
    return pool

# %%

# FLATTEN LAYER

def flatten(x_tensor):
    shaped = x_tensor.get_shape().as_list()
    reshaped = tf.reshape(x_tensor, [-1, shaped[1] * shaped[2] * shaped[3]])
    return reshaped

# %%

# FULLY CONNECTED LAYER:

def fully_conn(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal(shape=[x_tensor.get_shape().as_list()[1], num_outputs], mean=0.0, stddev=0.1)) 
    bias = tf.Variable(tf.zeros(shape=num_outputs))
    return tf.nn.relu(tf.matmul(x_tensor, weight) + bias)

# %%

# OUTPUT LAYER:
    
def weight_variable(x_tensor, num_outputs):
    initial = tf.truncated_normal(shape=[x_tensor.get_shape().as_list()[1], num_outputs], mean=0.0, stddev=0.1) 
    return tf.Variable(initial)

def bias_variable(num_outputs):
    return tf.Variable(tf.zeros(shape=num_outputs))

def output(x_tensor, num_outputs):
    logits1 = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    logits2 = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    logits3 = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    logits4 = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    logits5 = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    return [logits1, logits2, logits3, logits4, logits5]
    
# %%

# CREATE A CONVOLUTION MODEL METHOD

depth1 = 64
depth2 = 128
depth3 = 256
depth_full1 = 512
depth_full2 = 256
classes = 11


def conv_net(x, keep_prob):
    model = conv2d_maxpool(x, depth1, (3,3), (1,1), (2,2), (2,2))
    model = conv2d_maxpool(model, depth2, (3,3), (1,1), (2,2), (2,2))
    model = conv2d_maxpool(model, depth3, (3,3), (1,1), (2,2), (2,2))
    model = flatten(model)
    model = fully_conn(model, depth_full1)
    model = tf.nn.dropout(model, keep_prob)
    model = fully_conn(model, depth_full2)
    return output(model, classes)

# %%

# BUILD THE NEURAL NETWORK

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()


# Inputs
x = neural_net_image_input((32, 32, 1))
y = neural_net_label_input(6)
keep_prob = neural_net_keep_prob_input()


# Model
[logits1, logits2, logits3, logits4, logits5] = conv_net(x, keep_prob)


# Name logits Tensor, so that is can be loaded from disk after training
logits1 = tf.identity(logits1, name='logits1')
logits2 = tf.identity(logits2, name='logits2')
logits3 = tf.identity(logits3, name='logits3')
logits4 = tf.identity(logits4, name='logits4')
logits5 = tf.identity(logits5, name='logits5')


# Loss and Optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=y[:, 1])) + \
                          tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=y[: ,2])) + \
                          tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=y[: ,3])) + \
                          tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=y[: ,4])) + \
                          tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=y[: ,5]))

# %%

prediction = tf.stack([tf.nn.softmax(conv_net(x, 1)[0]),
                               tf.nn.softmax(conv_net(x, 1)[1]),
                               tf.nn.softmax(conv_net(x, 1)[2]),
                               tf.nn.softmax(conv_net(x, 1)[3]),
                               tf.nn.softmax(conv_net(x, 1)[4])])

prediction = tf.transpose(tf.argmax(prediction, 2))


# %%

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

# %%

# Accuracy
correct_pred = tf.equal(tf.to_int32(prediction), (y[:, 1:]))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# %%

# TRAINING METHOD

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    feed_dict = {
            x: feature_batch, 
            y: label_batch, 
            keep_prob: keep_probability}
    session.run(optimizer, feed_dict=feed_dict)

# %%
    
def print_stats(session, feature_batch, label_batch, loss, accuracy):
    current_cost = session.run(
        loss,
        feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    
    valid_accuracy = session.run(
        accuracy,
        feed_dict={x: valid_dataset, y: valid_labels, keep_prob: 1.})
    print('Loss: {:<8.3} Valid Accuracy: {:<5.3}'.format(
       current_cost,
        valid_accuracy))

# %%

epochs = 5
batch_size = 128
keep_probability = 0.7

# %%

save_model_path = './digit_reocgnition'

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    
    for epoch in range(epochs):
        offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
        batch_features = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2} '.format(epoch + 1), end='')
        print_stats(sess, batch_features, batch_labels, loss, accuracy)
    
    
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    
