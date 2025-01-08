#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 00:01:22 2023

@author: s
"""


import keras_tuner as kt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def build_model(hp):
    model = Sequential([
        Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(6,)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error')
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=4000,
                     hyperband_iterations=8)

tuner.search(X_train, Y_train, validation_data=(X_test, Y_test))

# Get optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The best number of units in the first dense layer: {best_hps.get('units')}")

print(f"The best learning rate for the Adam optimiser is: {best_hps.get('learning_rate')}")


'''


def build_model(hp):
    model = Sequential([
        Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(6,)),
        Dense(1)
    ])
    
    # Choose an optimiser
    optimiser_name = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    
    # Hyperparameters for Adam optimiser
    if optimiser_name == 'adam':
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=hp.Float('adam_learning_rate', 1e-4, 1e-2, sampling='log'))
    
    # Hyperparameters for SGD optimiser
    elif optimiser_name == 'sgd':
        optimiser = tf.keras.optimizers.SGD(
            learning_rate=hp.Float('sgd_learning_rate', 1e-4, 1e-2, sampling='log'),
            momentum=hp.Float('sgd_momentum', 0.5, 0.9, step=0.1))
    
    # Hyperparameters for RMSprop optimiser
    elif optimiser_name == 'rmsprop':
        optimiser = tf.keras.optimizers.RMSprop(
            learning_rate=hp.Float('rmsprop_learning_rate', 1e-4, 1e-2, sampling='log'))
    
    model.compile(optimizer=optimiser, loss='mean_squared_error')
    
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=10,
                     hyperband_iterations=8,
                     directory='my_dir', # Sub-directory
                     project_name='keras_tuning') # Project name

# Start hyperparameter search
tuner.search(X_train, Y_train, validation_data=(X_test, Y_test))

# Get optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The best number of units in the first dense layer: {best_hps.get('units')}")

print(f"The best learning rate for the Adam optimiser is: {best_hps.get('adam_learning_rate')}")
'''
