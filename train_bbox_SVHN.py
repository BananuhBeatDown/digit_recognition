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