import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization



def build_cnn(params):
    
    """
    
        INPUT:
        params - parameters to build cnn architecture
                
        OUTPUT:
        cnn - cnn model
        
    """    
    
    cnn = Sequential([
        Conv2D(params['conv_layer_size_1'], (params['kernel'], params['kernel']), activation=params['activation_conv'], input_shape=(params['input_size'], params['input_size'], params['channel']), padding=params['padding']),
        MaxPool2D(pool_size=(params['pool_size_1'], params['pool_size_1'])),
        Conv2D(params['conv_layer_size_2'], (params['kernel'], params['kernel']), activation=params['activation_conv'], padding=params['padding']),
        MaxPool2D(pool_size=(params['pool_size_1'], params['pool_size_1'])),
        Conv2D(params['conv_layer_size_3'], (params['kernel'], params['kernel']), activation=params['activation_conv'], padding=params['padding']),
        MaxPool2D(pool_size=(params['pool_size_2'], params['pool_size_2'])),
        Flatten(),
        Dense(params['dense_layer_1'], activation=params['activation_conv']),
        Dense(params['dense_layer_2'], activation=params['activation_conv']),
        Dropout(rate=params['dropout_rate']),
        Dense(params['dense_size'], activation=params['activation_dense'])
    ])
    return cnn



def train_cnn(cnn, params, train_dataset, val_dataset, y_train, y_val):
    
    """
    
        INPUT:
        img - Image matrix (list or numpy array)
        mutiplicator - mian numerical value for imager artifacts removal
                
        OUTPUT:
        img_filtered - Cleaned image (numpy array)
        
    """    
    
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    
    cnn.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    datagen = ImageDataGenerator(
        rescale=1/255,
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False, 
        zca_whitening=False,  
        rotation_range = 30, 
        zoom_range = 0.2, 
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        horizontal_flip = True,
        vertical_flip=False) 
    
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='loss',
        patience=3,
        verbose=1,
        factor=0.3,
        min_lr=0.000001
    )
    
    model_early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=params['patience'],
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False
    )

    history = cnn.fit(
        datagen.flow(
            train_dataset, 
            y_train, 
            batch_size=32, 
            shuffle=True, 
            seed=17), 
        validation_data=datagen.flow(
            val_dataset, 
            y_val, 
            batch_size=16,
            shuffle=True, 
            seed=17), 
        epochs=params['epochs'],
        callbacks = [learning_rate_reduction,
                model_early_stopping],
    )
    return cnn