#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:16:26 2017

@author: matthew_green
"""

from __future__ import print_function
from pickle_work_around import pickle_dump
from pickle_work_around import pickle_load
import matplotlib.pyplot as plt
import numpy as np
import random
import os


# load SVHN dataset without bounding boxes

pickle_file = 'pickles/SVHN_multi_bbox_32.pickle'

load = pickle_load(pickle_file)
s_train_dataset = load['train_dataset']
s_train_labels = load['train_labels']
s_valid_dataset = load['valid_dataset']
s_valid_labels = load['valid_labels']
s_test_dataset = load['test_dataset']
s_test_labels = load['test_labels']
del load

# %%

# load MNIST dataset

pickle_file = 'pickles/MNIST_multi_32.pickle'

load = pickle_load(pickle_file)
m_train_dataset = load['train_dataset']
m_train_labels = load['train_labels']
m_valid_dataset = load['valid_dataset']
m_valid_labels = load['valid_labels']
m_test_dataset = load['test_dataset']
m_test_labels = load['test_labels']
del load

# %%

#  Combine and randomize the datasets and their corresponding labels

def concat_data_and_labels(dataset1, dataset2, labels1, labels2):
    dataset_temp = np.concatenate((dataset1, dataset2), axis=0)
    labels_temp = np.concatenate((labels1, labels2), axis=0)
    idx = np.random.permutation(len(dataset_temp))
    dataset, labels = dataset_temp[idx], labels_temp[idx]
    return dataset, labels

train_dataset, train_labels = concat_data_and_labels(
        s_train_dataset, m_train_dataset, s_train_labels, m_train_labels)

valid_dataset, valid_labels = concat_data_and_labels(
        s_valid_dataset, m_valid_dataset, s_valid_labels, m_valid_labels)

test_dataset, test_labels = concat_data_and_labels(
        s_test_dataset, m_test_dataset, s_test_labels, m_test_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# %%

# Test display of combined images from the combined dataset

def displaySequence(n):
    plt.imshow(train_dataset[n].reshape(32, 32), cmap=plt.cm.Greys)
    plt.show()
    print ('Label : {}'.format(train_labels[n], cmap=plt.cm.Greys))

#display random sample to check if data is ok after creating sequences
displaySequence(random.randint(0, train_dataset.shape[0]))

# %%

# Save the datasets as individually
# labelled features in a pickle file. 

pickle_file = 'pickles/MS_combo_32.pickle'

save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }

pickle_dump(save, pickle_file)

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
