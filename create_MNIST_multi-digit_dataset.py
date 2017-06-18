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

# %%

# Synthetic Dataset Creator (1 to 5 digits per 64x64 image)

def synthetic_dataset_generator(dataset, labels, num_samples, new_size=64):
    
    trim_train_data = []
    [trim_train_data.append(i[:, 5:25]) for i in dataset]
    trim_train_data = np.array(trim_train_data)
    dataset = trim_train_data
    
    synthetic_dataset = np.ndarray(shape=(num_samples, new_size, new_size, 1))
    synthetic_labels = np.array([])
    data_labels = np.array([])   
    w = 0
    while w < num_samples:
        i = np.random.randint(1, 6)
        if i == 1:
            rand1 = np.random.randint(0, dataset.shape[0])
            
            filler = np.zeros(shape=(28, 4, 1)) - 0.5
            filled_train_data = np.array([np.hstack((filler, dataset[rand1], filler))])

            merged_train_data = np.ndarray(shape=(28, 28, 1), 
                                           dtype=np.float32)

            temp = np.hstack([filled_train_data[0]])
            merged_train_data[:, :, :] = temp
            temp_str = np.array([i, labels[rand1], 10, 10, 10, 10])
            data_labels = np.append(data_labels, temp_str)

            trial_train_dataset = []
            trial_train_dataset = np.reshape(merged_train_data, (28, 28 * 1))

            resized_train_dataset = np.ndarray(
                        shape=(new_size, new_size))

            temp = np.hstack(
                        [scipy.misc.imresize(trial_train_dataset, (new_size, new_size))])

            resized_train_dataset[:, :] = temp

            resized_train_dataset_4d = np.reshape(
                resized_train_dataset, (1, new_size, new_size, 1))

            synthetic_dataset[w, :, :, :] = resized_train_dataset_4d
            
            w += 1
            

        elif i == 2:
            rand1 = np.random.randint(0, dataset.shape[0])
            rand2 = np.random.randint(0, dataset.shape[0])
            
            new_train_data = np.concatenate((dataset[rand1:rand1+1], dataset[rand2:rand2+1]), axis=0)
            filler = np.zeros(shape=(6, 20, 1)) - 0.5
            filled_train_data = np.array([np.vstack((filler, i, filler)) for i in new_train_data])


            merged_train_data = np.ndarray(shape=(1, 40, 40, 1), 
                                           dtype=np.float32)

            temp = np.hstack([filled_train_data[0], filled_train_data[1]])
            merged_train_data[:, :, :] = temp
            temp_str = np.array([i, labels[rand1], labels[rand2], 10, 10, 10])
            data_labels = np.append(data_labels, temp_str)

            trial_train_dataset = []
            trial_train_dataset = np.reshape(merged_train_data, (40, 40 * 1))

            resized_train_dataset = np.ndarray(
                        shape=(new_size, new_size))

            temp = np.hstack(
                        [scipy.misc.imresize(trial_train_dataset, (new_size, new_size))])

            resized_train_dataset[:, :] = temp

            resized_train_dataset_4d = np.reshape(
                resized_train_dataset, (1, new_size, new_size, 1))

            synthetic_dataset[w, :, :, :] = resized_train_dataset_4d
            
            w += 1

            
        elif i == 3:
            rand1 = np.random.randint(0, dataset.shape[0])
            rand2 = np.random.randint(0, dataset.shape[0])
            rand3 = np.random.randint(0, dataset.shape[0])
                
            new_train_data = np.concatenate((dataset[rand1:rand1+1], dataset[rand2:rand2+1], 
                                                dataset[rand3:rand3+1]), axis=0)
            filler = np.zeros(shape=(16, 20, 1)) - 0.5
            filled_train_data = np.array([np.vstack((filler, i, filler)) 
                                          for i in new_train_data])

            merged_train_data = np.ndarray(shape=(60, 60, 1), 
                                           dtype=np.float32)

            temp = np.hstack([filled_train_data[0], filled_train_data[1], filled_train_data[2]])
            merged_train_data[:, :, :] = temp
            temp_str = np.array([i, labels[rand1], labels[rand2], labels[rand3], 10, 10])
            data_labels = np.append(data_labels, temp_str)

            trial_train_dataset = []
            trial_train_dataset = np.reshape(merged_train_data, (60, 60 * 1))

            resized_train_dataset = np.ndarray(
                        shape=(new_size, new_size))

            temp = np.hstack(
                        [scipy.misc.imresize(trial_train_dataset, (new_size, new_size))])

            resized_train_dataset[:, :] = temp

            resized_train_dataset_4d = np.reshape(
                resized_train_dataset, (1, new_size, new_size, 1))

            synthetic_dataset[w, :, :, :] = resized_train_dataset_4d
            
            w += 1

        elif i == 4:
            rand1 = np.random.randint(0, dataset.shape[0])
            rand2 = np.random.randint(0, dataset.shape[0])
            rand3 = np.random.randint(0, dataset.shape[0])
            rand4 = np.random.randint(0, dataset.shape[0])
            
            filled_train_data = np.concatenate((dataset[rand1:rand1+1], dataset[rand2:rand2+1], 
                                                dataset[rand3:rand3+1], dataset[rand4:rand4+1]), axis=0)
            filler = np.zeros(shape=(26, 20, 1)) - 0.5
            filled_train_data = np.array([np.vstack((filler, i, filler)) for i in filled_train_data])

            merged_train_data = np.ndarray(shape=(80, 80, 1), 
                                           dtype=np.float32)

            temp = np.hstack([filled_train_data[0], filled_train_data[1], filled_train_data[2],
                             filled_train_data[3]])
            merged_train_data[:, :, :] = temp
            temp_str = np.array([i, labels[rand1], labels[rand2], labels[rand3], labels[rand4], 10])
            data_labels = np.append(data_labels, temp_str)

            trial_train_dataset = []
            trial_train_dataset = np.reshape(merged_train_data, (80, 80 * 1))

            resized_train_dataset = np.ndarray(
                        shape=(new_size, new_size))

            temp = np.hstack(
                        [scipy.misc.imresize(trial_train_dataset, (new_size, new_size))])

            resized_train_dataset[:, :] = temp

            resized_train_dataset_4d = np.reshape(
                resized_train_dataset, (1, new_size, new_size, 1))

            synthetic_dataset[w, :, :, :] = resized_train_dataset_4d
            
            w += 1

        else:
            rand1 = np.random.randint(0, dataset.shape[0])
            rand2 = np.random.randint(0, dataset.shape[0])
            rand3 = np.random.randint(0, dataset.shape[0])
            rand4 = np.random.randint(0, dataset.shape[0])
            rand5 = np.random.randint(0, dataset.shape[0])
            
            filled_train_data = np.concatenate((dataset[rand1:rand1+1], dataset[rand2:rand2+1],
                                                dataset[rand3:rand3+1],dataset[rand4:rand4+1],
                                                dataset[rand5:rand5+1]), axis=0)

            filler = np.zeros(shape=(36, 20, 1)) - 0.5
            filled_train_data = np.array([np.vstack((filler, i, filler)) 
                                          for i in filled_train_data])
            
            merged_train_data = np.ndarray(shape=(100, 100, 1), 
                                           dtype=np.float32)

            temp = np.hstack([filled_train_data[0], filled_train_data[1], filled_train_data[2],
                             filled_train_data[3], filled_train_data[4]])
            merged_train_data[:, :, :] = temp
            temp_str = np.array([i, labels[rand1], labels[rand2], labels[rand3], labels[rand4], labels[rand5]])
            data_labels = np.append(data_labels, temp_str)

            trial_train_dataset = []
            trial_train_dataset = np.reshape(merged_train_data, (100, 100 * 1))

            resized_train_dataset = np.ndarray(
                        shape=(new_size, new_size))

            temp = np.hstack(
                        [scipy.misc.imresize(trial_train_dataset, (new_size, new_size))])

            resized_train_dataset[:, :] = temp

            resized_train_dataset_4d = np.reshape(
                resized_train_dataset, (1, new_size, new_size, 1))

            synthetic_dataset[w, :, :, :] = resized_train_dataset_4d
            
            w += 1

    # This belongs here
    synthetic_labels = np.reshape(data_labels, (-1, 6))

    return synthetic_dataset, synthetic_labels

# %%

# Create multi-digit train, valid, and test datasets and labels

train_dataset, train_labels = synthetic_dataset_generator(train_data, training_labels, 50000)
valid_dataset, valid_labels = synthetic_dataset_generator(train_data, training_labels, 9000)
test_dataset, test_labels = synthetic_dataset_generator(test_data, test_labels, len(test_data))

print(train_dataset.shape, train_labels.shape)
print(valid_dataset.shape, valid_labels.shape)
print(test_dataset.shape, test_labels.shape)

# %%

# Test display of the new images from the train_dataset

def displaySequence(n):
    plt.imshow(train_dataset[n].reshape(64, 64), cmap=plt.cm.Greys)
    plt.show()
    print ('Label : {}'.format(train_labels[n], cmap=plt.cm.Greys))

#display random sample to check if data is ok after creating sequences
displaySequence(random.randint(0, train_dataset.shape[0]))

# %%

# Save the datasets as individually
# labelled features in a pickle file. 

pickle_file = '/Users/matthew_green/Desktop/version_control/digit_recognition/metas/MNIST_multi_64.pickle'

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


