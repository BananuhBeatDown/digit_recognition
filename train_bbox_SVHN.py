#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:36:52 2017

@author: matthew_green
"""

from __future__ import print_function
from pickle_work_around import pickle_load
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import random

# Load the SVHN bounding box dataset

pickle_file = 'pickles/SVHN_multi_bbox_32.pickle'

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
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    
# display random sample to check if data is ok after creating sequences
# displaySequence_test(random.randint(0, train_dataset.shape[0]))
print('SVNH Bound Box Image Test:')
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

train_bbox = clean_train_bbox
valid_bbox = clean_valid_bbox
test_bbox = clean_test_bbox

del clean_train_bbox
del clean_valid_bbox
del clean_test_bbox

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
    bbox_pred = tf.matmul(x_tensor, weight_variable(x_tensor, num_outputs)) + bias_variable(num_outputs)
    return bbox_pred
    
# %%

# CREATE A CONVOLUTION MODEL METHOD

depth1 = 16
depth2 = 32
depth3 = 64
depth_full1 = 128
depth_full2 = 64
classes = 20


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
y = neural_net_bbox(20)
keep_prob = neural_net_keep_prob_input()


# Model
bbox_pred = conv_net(x, keep_prob)


# Name logits Tensor, so that is can be loaded from disk after training
bbox_pred = tf.identity(bbox_pred, name='bbox_pred')


# Loss
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, bbox_pred))))

# %%

# Bounding box intersection over union accuaracy

def bb_intersection_over_union(predictions, ground_truth):
    # determine the (x, y)-coordinates of the intersection rectangle
    iou_counter = 0
    for pred, gt in zip(predictions, ground_truth):     
        x1a = max(gt[1], pred[1])
        y1a = max(gt[0], pred[0])
        x1b = min(gt[3], pred[3])
        y1b = min(gt[2], pred[2])

        x2a = max(gt[5], pred[5])
        y2a = max(gt[4], pred[4])
        x2b = min(gt[7], pred[7])
        y2b = min(gt[6], pred[6])

        x3a = max(gt[9], pred[9])
        y3a = max(gt[8], pred[8])
        x3b = min(gt[11], pred[11])
        y3b = min(gt[10], pred[10])

        x4a = max(gt[13], pred[13])
        y4a = max(gt[12], pred[12])
        x4b = min(gt[15], pred[15])
        y4b = min(gt[14], pred[14])

        x5a = max(gt[17], pred[17])
        y5a = max(gt[16], pred[16])
        x5b = min(gt[19], pred[19])
        y5b = min(gt[18], pred[18])

        # compute the area of intersection rectangle
        interArea1 = (x1b - x1a + 1) * (y1b - y1a + 1)
        interArea2 = (x2b - x2a + 1) * (y2b - y2a + 1)
        interArea3 = (x3b - x3a + 1) * (y3b - y3a + 1)
        interArea4 = (x4b - x4a + 1) * (y4b - y4a + 1)
        interArea5 = (x5b - x5a + 1) * (y5b - y5a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box1AArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
        box1BArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)

        box2AArea = (gt[6] - gt[4] + 1) * (gt[7] - gt[5] + 1)
        box2BArea = (pred[6] - pred[4] + 1) * (pred[7] - pred[5] + 1)

        box3AArea = (gt[10] - gt[8] + 1) * (gt[11] - gt[9] + 1)
        box3BArea = (pred[10] - pred[8] + 1) * (pred[11] - pred[9] + 1)

        box4AArea = (gt[14] - gt[12] + 1) * (gt[15] - gt[13] + 1)
        box4BArea = (pred[14] - pred[12] + 1) * (pred[15] - pred[13] + 1)

        box5AArea = (gt[18] - gt[16] + 1) * (gt[19] - gt[17] + 1)
        box5BArea = (pred[18] - pred[16] + 1) * (pred[19] - pred[17] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou1 = interArea1 / float(box1AArea + box1BArea - interArea1)
        iou2 = interArea2 / float(box2AArea + box2BArea - interArea2)
        iou3 = interArea3 / float(box3AArea + box3BArea - interArea3)
        iou4 = interArea4 / float(box4AArea + box4BArea - interArea4)
        iou5 = interArea5 / float(box5AArea + box5BArea - interArea5)

        iou = np.mean([iou1, iou2, iou3, iou4, iou5])
        # return the intersection over union value
        if iou >= 0.5:
            iou_counter += 1
    return (iou_counter / float(len(ground_truth))) * 100

# %%

# Optimization algorithm

global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

# %%

# Loss and accuracy for training and validation datasets methods

def print_stats(session, feature_batch, label_batch, loss, bbox_pred):
    current_cost = session.run(
        loss,
        feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    
    train_pred_bbox = session.run(
            bbox_pred,
            feed_dict = {x: feature_batch, y: label_batch, keep_prob: 1.})
    
    valid_pred_bbox = session.run(
            bbox_pred,
            feed_dict = {x: valid_dataset, y: valid_bbox, keep_prob: 1.})
    
    print(' Loss: {:<8.3} Train bbox Accuracy: {:<5.3}% Valid bbox Accuracy: {:<5.3}%'.format(
       current_cost,
       bb_intersection_over_union(train_pred_bbox, label_batch),
       bb_intersection_over_union(valid_pred_bbox, label_batch)))

# %%
    
# Accuracy for testing dataset methods
    
def print_test_stats(session, bbox_pred):
    test_pred_bbox = session.run(
            bbox_pred,
            feed_dict = {x: test_dataset, y: test_bbox, keep_prob: 1.})
    
    print(' Test bbox Accuracy: {:<5.3}%'.format(
       bb_intersection_over_union(test_pred_bbox, test_bbox)))


# %%

# CNN training funciton

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    feed_dict = {
            x: feature_batch, 
            y: label_batch, 
            keep_prob: keep_probability}
    session.run(optimizer, feed_dict=feed_dict)

# %%

# Set CNN parameters

epochs = 1001
batch_size = 256
keep_probability = 0.9375

# %%

# Train the CNN

save_model_path = 'metas/my_model_bbox'

with tf.Session() as sess:
    # Initializing the variables
    
    sess.run(tf.global_variables_initializer())
    
    print("Initialized")
    
    for epoch in range(epochs):
        offset = (epoch * batch_size) % (train_bbox.shape[0] - batch_size)
        batch_features = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_bbox[offset:(offset + batch_size), :]
        train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        if epoch % 100 == 0:
            
            print('Epoch {:>2}'.format(epoch + 1), end='')
            print_stats(sess, batch_features, batch_labels, loss, bbox_pred)
         
    print_test_stats(sess, bbox_pred)   

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    print("Model save in file {}".format(save_path))

# %%

# Load real_test_dataset

pickle_file = 'pickles/real_test_dataset.pickle'

with open(pickle_file, 'rb') as f:
  load = pickle.load(f)
  real_test_dataset = load['real_test_dataset']
  del load  # hint to help gc free up memory
  print('Real test data and lables', real_test_dataset.shape)

# %%

# Image display test

def real_displaySequence(n):
    plt.imshow(real_test_dataset[n].reshape(32, 32), cmap=plt.cm.Greys)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")

#display random sample to check if data is ok after creating sequences
print('Real Dataset Image Test:')
real_displaySequence(random.randint(0, real_test_dataset.shape[0] - 1))


# %%

# Restore CNN model and make predicitons

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('metas/my_model_bbox.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('metas'))
    print("Model restored.")  
    

    print("Initialized")
    # test_prediction = sess.run(test_prediction, feed_dict={tf_test_dataset : real_test_dataset})
    bbox_prediction = sess.run(bbox_pred, feed_dict={x : real_test_dataset, keep_prob: 1.})

# %%

# Display the real test images and the predicted bounding boxes

for n in range(6):
    fig,ax=plt.subplots(1)
    plt.imshow(real_test_dataset[n].reshape(32, 32), cmap=plt.cm.Greys)
    
    rect1 = patches.Rectangle((bbox_prediction[n][1], bbox_prediction[n][0]),
                             bbox_prediction[n][3], bbox_prediction[n][2],
                             linewidth=1,edgecolor='r',facecolor='none')
    
    rect2 = patches.Rectangle((bbox_prediction[n][5], bbox_prediction[n][4]),
                             bbox_prediction[n][7], bbox_prediction[n][6],
                             linewidth=1,edgecolor='r',facecolor='none')
    
    rect3 = patches.Rectangle((bbox_prediction[n][9], bbox_prediction[n][8]),
                             bbox_prediction[n][11], bbox_prediction[n][10],
                             linewidth=1,edgecolor='r',facecolor='none')
    
    rect4 = patches.Rectangle((bbox_prediction[n][13], bbox_prediction[n][12]),
                             bbox_prediction[n][15], bbox_prediction[n][14],
                             linewidth=1,edgecolor='r',facecolor='none')
    
    rect5 = patches.Rectangle((bbox_prediction[n][17], bbox_prediction[n][16]),
                             bbox_prediction[n][19], bbox_prediction[n][18],
                             linewidth=1,edgecolor='r',facecolor='none')
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.add_patch(rect5)
    
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")