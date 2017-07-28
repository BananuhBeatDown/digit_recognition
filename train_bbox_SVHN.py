#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:36:52 2017

@author: matthew_green
"""

from __future__ import print_function
from pickle_work_around import pickle_load
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import random

pickle_file = 'metas/SVHN_multi_bbox_32.pickle'

load = pickle_load(pickle_file)
train_dataset = load['train_dataset']
train_bbox = load['train_bbox']
valid_dataset = load['valid_dataset']
valid_bbox = load['valid_bbox']
test_dataset = load['test_dataset']
test_bbox = load['test_bbox']

del load  # hint to help gc free up memory

print('Training set', train_dataset.shape, train_bbox.shape)
print('Validation set', valid_dataset.shape, valid_bbox.shape)
print('Test set', test_dataset.shape, test_bbox.shape)

# %%
    
# DIGIT AND BOUNDING BOX DISPLAY TEST:

def displaySequence_test(n):
    fig,ax=plt.subplots(1)
    plt.imshow(train_dataset[n].reshape(32, 32), cmap=plt.cm.Greys)
    
    for i in np.arange(4):
        rect = patches.Rectangle((train_bbox[n][1][i], train_bbox[n][0][i]),
                                  train_bbox[n][3][i], train_bbox[n][2][i],
                                  linewidth=1,edgecolor='r',facecolor='none')
        
        ax.add_patch(rect)
    plt.show
    
    print(n)
    
# display random sample to check if data is ok after creating sequences
# displaySequence_test(random.randint(0, train_dataset.shape[0]))
displaySequence_test(random.randint(0, train_dataset.shape[0] - 1))

# %%

# RESIZE BOUNDING BOX ARRAYS INTO Nx20

np.set_printoptions(suppress=True, precision=3)

clean_train_bbox = []
clean_valid_bbox = []
clean_test_bbox = []

for i in train_bbox:
    clean_train_bbox.append(i.reshape(20, 1, order='F'))
clean_train_bbox = np.array(clean_train_bbox)
clean_train_bbox = np.reshape(clean_train_bbox, (-1, 20))

for j in valid_bbox:
    clean_valid_bbox.append(j.reshape(20, 1, order='F'))
clean_valid_bbox = np.array(clean_valid_bbox)
clean_valid_bbox = np.reshape(clean_valid_bbox, (-1, 20))

for k in test_bbox:
    clean_test_bbox.append(k.reshape(20, 1, order='F'))
clean_test_bbox = np.array(clean_test_bbox)
clean_test_bbox = np.reshape(clean_test_bbox, (-1, 20))

print(clean_train_bbox.shape, clean_valid_bbox.shape, clean_test_bbox.shape)

# %%

# INPUT LAYER:

def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_bbox(n_classes):
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
    bbox_pred = (x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    return bbox_pred
    
# %%

# CREATE A CONVOLUTION MODEL METHOD

depth1 = 16
depth2 = 32
depth3 = 64
depth_full1 = 128
depth_full2 = 64
classes = 11


def conv_net(x, keep_prob):
    model = conv2d_maxpool(x, depth1, (5,5), (1,1), (2,2), (2,2))
    model = conv2d_maxpool(model, depth2, (3,3), (1,1), (2,2), (2,2))
    model = conv2d_maxpool(model, depth3, (3,3), (1,1), (2,2), (2,2))
    model = flatten(model)
    model = fully_conn(model, depth_full1)
    model = tf.nn.dropout(model, keep_prob)
    model = fully_conn(model, depth_full2)
    return output(model, classes)
