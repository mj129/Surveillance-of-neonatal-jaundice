Surveillance of neonatal jaundice using sternal skin images via deep learning

# Overview

This code uses convolutional neural network to estimate neonatal jaundice using sternal skin images.

# Contents

imgs: 12 skin images of 4 newborns with labels ('imgs/labels.csv'), and the prediction results of these images ('imgs/results.csv').
model: The trained weight of our network ('model.h5').
regression.py: The train code.
test.py: The test code.
utils.py: Some functions.

# System Requirements

## Hardware Requirements

The proposed network requires a computer with GPU to run the convolutional neural network. 
For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
GPU: NVIDIA 1080Ti

## OS Requirements

The code has been tested on Linux operating systems.

## Package dependencies

Users should install the following python packages prior to run the code:

python3.5
tensorflow-gpu
keras
pandas
numpy
opencv

# Instructions
Since we have provided the trained weight of the proposed network in folder 'model', 
you can run 'test.py' directly to estimate the bilirubin value of images in folder 'imgs'.
We also provide the prediction results of 'test.py' for these images in 'imgs/results.csv'.

The 'regression.py' is used to train the proposed network, but the training data is too large to 
provide, so 'regression.py' may not run in your computer.

We will put the code in an open source repository soon. 