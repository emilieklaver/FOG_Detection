"""
Convolution neural network created for the detection of freezing of gait 
by I.B. Heijink, adapted by E.C. Klaver.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
import tensorflow_addons as tfa
from tensorflow_addons import optimizers


def createCNN1D(shape, filter_length):        # filter_length=6 in script
    """ Creates CNN according to architecture as described in master thesis Irene Heijink.
    Input:
    1. shape, the shape of X
    2. filter_length, size of convolution kernel
    Output: model, a CNN that can be trained and tested"""
    model = keras.Sequential()  # simple sequential model
    model.add(layers.Conv1D(8, kernel_size = filter_length, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(layers.MaxPooling1D())
    model.add(Dropout(0.2))

    model.add(layers.Conv1D(16, kernel_size = 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(Dropout(0.2))  

    model.add(layers.Conv1D(32, kernel_size = 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(layers.Flatten())
    model.add(Dropout(0.2))


    model.add(layers.Dense(10, activation='relu'))  
    model.add(Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
    loss="binary_crossentropy",
    optimizer=tfa.optimizers.AdamW(weight_decay=0.001),
    metrics=["accuracy"],
    )
    
    print(model.summary())
    return model