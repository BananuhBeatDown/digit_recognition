#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:58:28 2017

@author: matthew_green
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:22:25 2017

@author: Matt Green
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves.urllib.request import urlretrieve
from pickle_work_around import pickle_dump
import random
import scipy
import gzip

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/Users/matthew_green/Desktop/version_control/digit_recognition'

def maybe_download(filename):
  """A helper to download the data files if not present."""
  if not os.path.exists(WORK_DIRECTORY):
    os.mkdir(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  else:
    print('Already downloaded', filename)
  return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

# %%

IMAGE_SIZE = 28
PIXEL_DEPTH = 255

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  
  For MNIST data, the number of channels is always 1.

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    # Skip the magic number and dimensions; we know these values.
    bytestream.read(16)
    
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

train_data = extract_data(train_data_filename, 60000)
test_data = extract_data(test_data_filename, 10000)

# %%

def extract_labels(filename, num_images):
  """Extract the labels."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    # Skip the magic number and count; we know these values.
    bytestream.read(8)
    
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8)
  return labels

training_labels = extract_labels(train_labels_filename, 60000)
test_labels = extract_labels(test_labels_filename, 10000)