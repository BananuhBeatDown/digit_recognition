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
train_bbox = load['train_bbox']
valid_dataset = load['valid_dataset']
valid_labels = load['valid_labels']
valid_bbox = load['valid_bbox']
test_dataset = load['test_dataset']
test_labels = load['test_labels']
test_bbox = load['test_bbox']
del load  # hint to help gc free up memory
print('Training set', train_dataset.shape, train_labels.shape, train_bbox.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape, valid_bbox.shape)
print('Test set', test_dataset.shape, test_labels.shape, test_bbox.shape)

# %%
    
# DIGIT AND BOUNDING BOX DISPLAY TEST:

def displaySequence_test(n):
    fig,ax=plt.subplots(1)
    plt.imshow(train_dataset[n].reshape(64, 64), cmap=plt.cm.Greys)
    
    for i in np.arange(4):
        rect = patches.Rectangle((train_bbox[n][1][i], train_bbox[n][0][i]),
                                  train_bbox[n][3][i], train_bbox[n][2][i],
                                  linewidth=1,edgecolor='r',facecolor='none')
        
        ax.add_patch(rect)
                                 
            
    plt.show
    print ('Label : {}'.format(train_labels[n], cmap=plt.cm.Greys), n)
    print(n)
# display random sample to check if data is ok after creating sequences
# displaySequence_test(random.randint(0, train_dataset.shape[0]))
displaySequence_test(random.randint(0, train_dataset.shape[0] - 1))

# %%

# FORMAT BOUNDING BOXES:

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


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_bbox_input():
    return tf.placeholder(tf.float32, shape=(None, 20), name='bbox')


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

