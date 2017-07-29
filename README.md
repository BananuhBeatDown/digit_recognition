# Convolutional Neural Network Digit Recognition and Bounding Box Prediction

This program performs image recognition and bounding box recognition on a series of digtits ranging in squence from 1-5. comprised from  and  datasets using a Convolutional Neural Network (CNN), and also also perform bounding prediciton . The images from the [MNIST](http://yann.lecun.com/exdb/mnist/) and [SVHN](http://ufldl.stanford.edu/housenumbers/) datasets are preprocessed and normalized then used to train a CNN consisting of convolutional, max pooling, dropout, fully connected, and output layers.

## Install

- Python 3.6
    + I recommend installing [Anaconda](https://www.continuum.io/downloads) as it is alreay set up with standard machine learning libraries
    + If unfamiliar with the command line there are graphical installs for macOS, Windows, and Linux
- [PIL](http://www.pythonware.com/products/pil/)
    + `pip install pillow` for python 3
- [six](https://pythonhosted.org/six/)
- [TensorFlow](https://www.tensorflow.org/install/?nav=true)

## Dataset

In this study the MNIST and SVHN datasets were used to create a combined dataset of hand drawn digits and house numbers in groupings of 1-5 digits. There are a total of roughly 320k images: 280k training images, 15k validation images, and 23k testing images. 

The images are 32x32x1 grayscale format with 32 representing the pixel width and height and 1 representing the gray color dimension. Each image has a corresponding label which lists the numbers of digits in the image and digit themselves, including a label representing the absence of a digit in cases where there are less than 5 digits (the maximum number of digits in an image). The SVHN dataset also includes bounding box information which will be used in the second half of the project to determine digit location.

## Parameters

`depth` - Alter the depths of the CNN layers using common memory sizes
`epochs` - number of training iterations
`batch_size` - set to highest number your machine has memory for during common memory sizes
`keep_probability` - probability of keeping activation node in dropout layer

## Example Output

Run the files in the order specified below.

**Command Line**

`python create MNIST_multi-digit-dataset.py`

- Creates multi-digit MNIST 32x32 dataset

<img src="https://user-images.githubusercontent.com/10539813/28746962-04d790f0-7495-11e7-85cc-1f4337941742.png", width="512">

`python create_bbox_SVHN_dataset.py` 

- Creates SVHN 32x32 dataset with bounding boxes

<img src="https://user-images.githubusercontent.com/10539813/28746964-0b39fb22-7495-11e7-93c6-7eb682c58720.png", width="512">

`python create_combined_dataset.py`

- Combines and randomizes the previous two dataset

<img src="https://user-images.githubusercontent.com/10539813/28746966-0c37c464-7495-11e7-9749-3fc294d59cb6.png" width="512">

`python create_real_world_dataset.py`

- Create a grayscaled images from real world pictures

<img src="https://user-images.githubusercontent.com/10539813/28746967-0d71c0e6-7495-11e7-8b95-d0c0d67913ce.png", width="512">

`python train_digit_recognition_CNN.py`

- Trains network on the combined dataset and outputs loss and accuracy data into tensorboard files

<img src="https://user-images.githubusercontent.com/10539813/28746968-0ed0d3aa-7495-11e7-84c7-c8daf0a09e9c.png", width="512">

**To view the tensorboard loss and accuracy outputs, follow [these instruntions](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard) from the tensorflow website.**

`train_bounding_box_CNN.py`

- Trains the network on the SVHN bounding box dataset and outputs predicted bounding box examples on the real world dataset

<img src="https://user-images.githubusercontent.com/10539813/28746969-102ab5ae-7495-11e7-984f-858cbc5b6783.png", width="512">

<img src="https://user-images.githubusercontent.com/10539813/28746999-9693646a-7495-11e7-84b1-84b1b3ed261c.png", width="256">

## License
The image_classification program is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). I encourage you to use it, and enhance your understanding of CNNs and the deep learning concepts therein. :)

