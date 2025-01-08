#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:07:58 2023

@author: s
"""
import numpy as np
import pandas as pd
import tensorflow as tf


#define csv file to extract from
#gdrive csvFile = r'C:/Users/s5230048/OneDrive - Griffith University/MIKE ZERO/Extracted (csv) Data/AllRuns_SingleTime.csv'
csvFile = "/Users/s/PhD/AllRuns_SingleTime_Train.csv"
testFile = "/Users/s/PhD/AllRuns_SingleTime_Test.csv"
midFile = "/Users/s/PhD/avg of mid points.csv"
testmidFile = "/Users/s/PhD/test avg of mid points.csv"

#create dataframe
df = pd.read_csv(csvFile)

#remove unwatned row labels in first column
columns = df.columns # get dataframe column header names
columns = columns[1:244] #remove the header from the first column 

# sample the df and convert to numpy array
X_train = df[columns].iloc[2:8].values.astype(float).T  
# the "df[columns]" selects the columns specified in the columns array from above;
# the ".iloc[3:8]" selects rows 3 to 8 which correspond to relevant SBW and wave parameters in the csv file
# the ".values.astype(float)" converts the df to a numpy array of floats


#data frame for average of mid 3, 5, and 7 points in shoreline change to help reduce noise in output
dfmid = pd.read_csv(midFile)

#Turn on one line to create output with avg of mid 3, 5, 7 or absolute mid (this currently doesn't have equivalent test set data)

#Y_train = dfmid[columns].iloc[0].values.astype(float).T #absolute mid point
Y_train = dfmid[columns].iloc[5].values.astype(float).T # avg of mid 3
#Y_train = dfmid[columns].iloc[12].values.astype(float).T # avg of mid 5
#Y_train = dfmid[columns].iloc[21].values.astype(float).T # avg of mid 7



#create nn model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    #tf.keras.layers.Dense(6, activation='sigmoid'),
    tf.keras.layers.Dense(320, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh'),
    tf.keras.layers.Dense(1)  #output layer with 1var
])
#optimizer
opt = tf.keras.optimizers.Adam(learning_rate = 0.0066, clipnorm = 1)
#compile the model
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

#model.fit(X_train, Y_train, epochs=100, batch_size=1)

model.summary()


#validation
dftest = pd.read_csv(testFile)
dftestmid = pd.read_csv(testmidFile)

columns_test = dftest.columns # get dataframe column header names
columns_test = columns_test[1:40] #remove the header from the first column 
columns_testmid = dftestmid.columns
columns_testmid = columns_testmid[1:40]

X_test = dftest[columns_test].iloc[2:8].values.astype(float).T  

#Y_test = dftest[columns_test].iloc[15].values.astype(float) # row 12 is the max shoreline row
Y_test = dftestmid[columns_testmid].iloc[5].values.astype(float).T # entire shoreline output

#train the model
#model.fit(X_train, Y_train, epochs=100, batch_size=1, validation_data=(X_test, Y_test))

model.evaluate(X_test,  Y_test, verbose=2)

# Train the model and save the history
history = model.fit(X_train, Y_train, epochs=420, batch_size=4, validation_data=(X_test, Y_test))

# Plot training & validation loss values
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=4)
print('test loss, test mse:', results)

# Make predictions
print('\n# Predictions')
predictions = model.predict(X_test)

#order of variables = ([[ wave height, wave period, wave angle, dist from shore, lenght, width]])
# Make predictions
input_data = np.array([[1.5, 10, 180, 350, 100, 50]])
predictions = model.predict(input_data)
# Evaluate the model if needed
mse = np.mean((predictions - Y_test) ** 2)
print(f"Mean Squared Error: {mse}")
print(predictions)


# You can also use the model for predictions on new data
new_data = np.array([[1.5, 10, 180, 20, 100, 50]])
new_predictions = model.predict(new_data)
print("Predictions for new data:")
print(new_predictions)




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
                     max_epochs=10,
                     hyperband_iterations=8)

tuner.search(X_train, Y_train, validation_data=(X_test, Y_test))

# Get optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The best number of units in the first dense layer: {best_hps.get('units')}")

print(f"The best learning rate for the Adam optimiser is: {best_hps.get('learning_rate')}")





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

