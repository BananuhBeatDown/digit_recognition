#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:31:39 2017

@author: matthew_green
"""


from __future__ import print_function
import os
from PIL import Image
import numpy as np
from six.moves import cPickle as pickle



rootdir = 'rgb_house_numbers'
newdir = 'gray_house_numbers'

if not os.path.exists(newdir):
    os.makedirs(newdir)

i = 1
for  subdir, dirs, pics in os.walk(rootdir): 
    for pic in pics:
        if os.path.exists(
            'gray_house_numbers/gray_num_{}.png'.format(i)):
            print('Image gray_num_{}.png already exists!'.format(i))
        else:
            img = Image.open('{}/{}'.format(rootdir, pic)).convert('L')
            img.save('{}/gray_num_{}.png'.format(newdir, i))
            print('Creating gray_num_{}.png created!'.format(i))
        i += 1


real_test_dataset = np.ndarray(shape=(6, 32, 32, 1), dtype=np.float32)
real_test_labels = np.array([[1, 6, 1, 10, 10], [2, 5, 10, 10, 10], [1,8, 0, 10, 10], 
                             [5, 9, 10, 10, 10], [6, 5, 10, 10, 10], [3, 0, 6, 0, 10]])

w = 0

for  subdir, dirs, pics in os.walk(newdir): 
    for pic in pics:
        img = Image.open('gray_house_numbers/{}'.format(pic))
        pixels = []
        pixels = np.array(img.getdata())
        pixels = (pixels - 128.) / 128.
        pixels = pixels.reshape(1, 32, 32, 1)
        real_test_dataset[w, :, :, :] = pixels
        w += 1
print(real_test_dataset.shape)
print(real_test_labels.shape)

# %%

pickle_file = 'pickles/real_test_dataset.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'real_test_dataset': real_test_dataset,
    'real_test_labels': real_test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)