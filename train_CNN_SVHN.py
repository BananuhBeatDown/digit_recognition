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