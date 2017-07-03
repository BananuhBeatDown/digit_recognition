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

path = '/Users/matthew_green/Desktop/version_control/digit_recognition/metas'
os.chdir(path)

pickle_file = 'SVHN_multi_bbox_64.pickle'

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
    plt.imshow(train_dataset[n].reshape(64, 64), cmap=plt.cm.Greys)
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
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')

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
x = neural_net_image_input((64, 64, 1))
y = neural_net_label_input(11)
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


global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 100000, 0.95)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)


# Accuracy
correct_pred = tf.equal([logits1, logits2, logits3, logits4, logits5], [y[1], y[2], y[3], y[4], y[5]])
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# %%

# TRAINING METHOD

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch, bbox_batch):
    feed_dict = {
            x: feature_batch, 
            y: label_batch, 
            keep_prob: keep_probability}
    session.run(optimizer, feed_dict=feed_dict)

# %%
    
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    current_cost = session.run(
        cost,
        feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    valid_accuracy = session.run(
        accuracy,
        feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})
    print('Loss: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        current_cost,
        valid_accuracy))
    
# %%

# %%

steps = 20000

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        batch_bbox = clean_train_bbox[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_train_bbox: batch_bbox}
        _, l, predictions, bbox_preds = sess.run(
            [optimizer, loss, train_prediction, train_bbox_pred], feed_dict=feed_dict)
        if step % 200 == 0:
            print('Minibatch loss at step {}: {:.3f}'.format(step, l))
            print('Minibatch accuracy: {:.3f}%'.format(accuracy(predictions, batch_labels[:, 1:6])))
            print('Minibatch Bounding Box accuracy: {:.3f}%'.format(
                    bb_intersection_over_union(bbox_preds, batch_bbox)))
            print('Validation accuracy: {:.3f}%'.format(accuracy(valid_prediction.eval(), valid_labels[:, 1:6])))
            print('Validation Bounding Box accuracy: {:.3f}%'.format(
                           bb_intersection_over_union(valid_bbox_pred.eval(), clean_valid_bbox)))
    print('Test accuracy: {0:.3f}%'.format(accuracy(test_prediction.eval(), test_labels[:, 1:6])))
    print('Test Bounding Box accuracy: {:.3f}%'.format(
                    bb_intersection_over_union(test_bbox_pred.eval(), clean_test_bbox)))
    
    # Save the variables to disk
    save_path = saver.save(sess, "C:/Users/Matt Green/Desktop/digit_recognition/metas/bbox_model")
    print("Model save in file: {}".format(save_path))    
    
    
