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



