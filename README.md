# Faster Automated annotation of cellular cryo-electron tomograms using convolutional neural network

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and based on the previous work [Convolutional neural networks for automated annotation of cellular cryo-electron tomograms](https://www.nature.com/articles/nmeth.4405)

---

## Overview

### Data

TEM image stack of [PC12 cell](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-8594), a tomogram after binning 4 is 864*864*94 pixels, 28 angstrom per pixel. There are four features to annotate, microtubules, ribosomes, single membranes and double membranes. 

### Data augmentation

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.


### Training

Loss function for the training is basically just a binary crossentropy.


---

## How to use

The full U-Net will be implemented in [EMAN2](https://blake.bcm.edu/emanwiki/EMAN2). And you can have an idea of how the previous 4-layer CNN works [here](https://blake.bcm.edu/emanwiki/EMAN2/Programs/tomoseg).

### Dependencies

Follow the installation of EMAN2.

Also, this code should be compatible with Python versions 2.7.14.


### Results

