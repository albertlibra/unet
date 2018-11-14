import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras



def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #print(conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #print(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #print(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #print(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
   # print(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #print(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
   # print(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #print(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #print(conv4)
    drop4 = Dropout(0.5)(conv4)
    #print(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #print(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #print(conv5)
    drop5 = Dropout(0.5)(conv5)
    #print(drop5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
<<<<<<< HEAD
    #print(up6)
    merge6 = add([drop4,up6])
    #print(merge6)
=======
    merge6 = Concatenate(axis=3)([drop4,up6])
>>>>>>> 59ecb295bbb4218b2dda7dfc4abe5fd4e0ef224c
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #print(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
<<<<<<< HEAD
    #print(up7)
    merge7 = add([conv3,up7])
=======
    merge7 = Concatenate(axis=3)([conv3,up7])
>>>>>>> 59ecb295bbb4218b2dda7dfc4abe5fd4e0ef224c
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
<<<<<<< HEAD
    merge8 = add([conv2,up8])
=======
    merge8 = Concatenate(axis=3)([conv2,up8])
>>>>>>> 59ecb295bbb4218b2dda7dfc4abe5fd4e0ef224c
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
<<<<<<< HEAD
    merge9 = add([conv1,up9])
=======
    merge9 = Concatenate(axis=3)([conv1,up9])
>>>>>>> 59ecb295bbb4218b2dda7dfc4abe5fd4e0ef224c
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(6, 1, activation = 'softmax')(conv9)

    # create the model
    model = Model(input = inputs, output = conv10)

    # compile the model
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    #print details of the layers with size of inputs/outputs in a table.
    model.summary()


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


