#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:47:45 2017

@author: matthew_green
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import tarfile

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random

path = '/Users/matthew_green/Desktop/version_control/digit_recognition'
os.chdir(path)

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  else:
    print("Already downloaded")
  return filename

train_filename = maybe_download('train.tar.gz')
test_filename = maybe_download('test.tar.gz')
extra_filename = maybe_download('extra.tar.gz')

# %%

np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = root
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)

# %%

import h5py

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
# %%

fin = os.path.join(train_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
train_data = dsf.getAllDigitStructure_ByDigit()

fin = os.path.join(test_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
test_data = dsf.getAllDigitStructure_ByDigit()

fin = os.path.join(extra_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
extra_data = dsf.getAllDigitStructure_ByDigit()

    
# %%

# Remove 6-digit pic and bad-pic from dataset
train_data = np.delete(train_data, 29929, axis=0) # 6-digits

extra_data = np.delete(extra_data, 43926, axis=0) # bad-pic

# %%

train_folders = 'train'
test_folders = 'test'
extra_folders = 'extra'

np.set_printoptions(suppress=True)
import PIL.Image as Image
image_size = 64



def generate_dataset(data, folder):
    
    bbox_dataset = np.ndarray([len(data), 4, 5], dtype='float32')
    dataset = np.ndarray([len(data),image_size,image_size,1], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) * 10
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        if os.path.isfile(os.path.join(folder, filename)) == False:
            pass
        else:
            fullname = os.path.join(folder, filename)
            im = Image.open(fullname)
            boxes = data[i]['boxes']
            num_digit = len(boxes)
            
            labels[i,0] = num_digit
            top = np.ndarray([num_digit], dtype='float32')
            left = np.ndarray([num_digit], dtype='float32')
            height = np.ndarray([num_digit], dtype='float32')
            width = np.ndarray([num_digit], dtype='float32')
            for j in np.arange(num_digit):
                if j < 5: 
                    labels[i,j+1] = boxes[j]['label']
                    if boxes[j]['label'] == 10: labels[i,j+1] = 0
                else: print('#',i,'image has more than 5 digits.')
                top[j] = boxes[j]['top']
                
                left[j] = boxes[j]['left']
                height[j] = boxes[j]['height']
                width[j] = boxes[j]['width']
    
            im_btop = np.amin(top)
            im_bleft = np.amin(left)
            im_bheight = np.amax(top) + height[np.argmax(top)] - im_btop
            im_bwidth = np.amax(left) + width[np.argmax(left)] - im_bleft
            
            im_top = im_btop - 0.15 * im_bheight
            im_left = im_bleft - 0.15 * im_bwidth
            im_bottom = np.amin([(im_btop + 1.15 * im_bheight), im.size[1]])
            im_right = np.amin([(im_bleft + 1.15 * im_bwidth), im.size[0]])
            # print(im_top, im_left, im_bottom, im_right)
            # print(im.size)
            
            top_ratio = top - im_top
            left_ratio = left - im_left
            top_ratio = top_ratio / (im_bottom - im_top) * image_size
            left_ratio = left_ratio / (im_right - im_left) * image_size
            
            height_ratio = (height / (im_bottom - im_top)) * image_size
            width_ratio = (width / (im_right - im_left)) * image_size
        
            
            # print(top_ratio, left_ratio, height_ratio, width_ratio)
            bbox = [top_ratio, left_ratio, height_ratio, width_ratio]
            bbox_single = np.ones(shape=(4, 5)) * -15.
            bbox_single[2:] = bbox_single[2:] + 25.
            for k in np.arange(4):
                if type(bbox[k]) == float:
                    bbox_single[k][0] = bbox[k]
                else:
                    while len(bbox[k]) < 5:
                        bbox[k] = np.append(bbox[k], -15.)
            if bbox_single[0].sum() != -75.:
                bbox = bbox_single
            else:
                bbox = np.array(bbox)
            for l in range(1, 5):
                if bbox[2][l] == -15.0:
                    bbox[2][l] = 10.0
                if bbox[3][l] == -15.0:
                    bbox[3][l] = 10.0
            
            # print(bbox)
            bbox_dataset[i, :, :] = bbox[:, :]
            # print(bbox_dataset[i])
            
            im = im.crop((int(np.floor(im_left)), int(np.floor(im_top)), int(np.floor(im_right)),
                          int(np.floor(im_bottom)))).resize([image_size, image_size], Image.ANTIALIAS)
            im = np.dot(np.array(im, dtype='float32'), [[0.2989],[0.5870],[0.1140]]) # Changes RGB to grey-scale
            mean = np.mean(im, dtype='float32')
            std = np.std(im, dtype='float32', ddof=1)
            if std < 1e-4: std = 1.
            im = (im - mean) / std
            dataset[i,:,:,:] = im[:,:,:]
        
    return dataset, labels, bbox_dataset

train_dataset, train_labels, train_bbox = generate_dataset(train_data, train_folders)
print(train_dataset.shape, train_labels.shape, train_bbox.shape)

test_dataset, test_labels, test_bbox = generate_dataset(test_data, test_folders)
print(test_dataset.shape, test_labels.shape, test_bbox.shape)

extra_dataset, extra_labels, extra_bbox = generate_dataset(extra_data, extra_folders)
print(extra_dataset.shape, extra_labels.shape, extra_bbox.shape)

# %%

def displaySequence_test(n):
    fig,ax=plt.subplots(1)
    plt.imshow(test_dataset[n].reshape(64, 64), cmap=plt.cm.Greys)
    
    for i in np.arange(4):
        rect = patches.Rectangle((test_bbox[n][1][i], test_bbox[n][0][i]),
                                  test_bbox[n][3][i], test_bbox[n][2][i],
                                  linewidth=1,edgecolor='r',facecolor='none')
        
        ax.add_patch(rect)                               
    plt.show
    
    print ('Label : {}'.format(test_labels[n], cmap=plt.cm.Greys), n)
    print(n)
    
# display random sample to check if data is ok after creating sequences
displaySequence_test(random.randint(0, test_dataset.shape[0] - 1))

# %%

# Create new training and valid datasets by mixing original training and valid datasets
random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,1] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,1] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,1] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,1] == (i))[0][200:].tolist())
    
    
random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)


valid_bbox = np.concatenate((extra_bbox[valid_index2,:,:], train_bbox[valid_index,:,:]), axis=0)
valid_dataset = np.concatenate((extra_dataset[valid_index2,:,:,:], train_dataset[valid_index,:,:,:]), axis=0)
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)
train_bbox_t = np.concatenate((extra_bbox[train_index2,:,:], train_bbox[train_index,:,:]), axis=0)
train_dataset_t = np.concatenate((extra_dataset[train_index2,:,:,:], train_dataset[train_index,:,:,:]), axis=0)
train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)



print(train_dataset_t.shape, train_labels_t.shape, train_bbox_t.shape)
print(test_dataset.shape, test_labels.shape, test_bbox.shape)
print(valid_dataset.shape, valid_labels.shape, valid_bbox.shape)
