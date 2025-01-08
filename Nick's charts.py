# -*- coding: utf-8 -*-
"""
SLmodel.py

First pass attempt at developing an ML model for Sub's shoreline data

nc, nov, 2023

"""

#%%

TrainingDataFile = "AllRuns_SingleTime_Train.csv"
TestDataFile = "AllRuns_SingleTime_Test.csv"

Nepochs = 4000
Nneurons_Input = 10
verbose = 0 # 1 => display model training/prediction progress data to console or 0 => don't

#%%

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

import tensorflow as tf

#%% custom function to compute skill metrics

def getSkillMetrics(actual_values, predicted_values):
    
    metrics_list = [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanAbsolutePercentageError(),
        tf.keras.metrics.RootMeanSquaredError(),
    ]
    
    # Dictionary to store computed metrics
    computed_metrics = {}
    
    # Loop over metrics and compute each metric
    for metric in metrics_list:
        metric.update_state(actual_values, predicted_values)
        computed_metric_value = metric.result().numpy()
        computed_metrics[metric.name] = computed_metric_value
    
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(list(computed_metrics.items()), columns=['Metric', 'Value'])
    
    return metrics_df

#%% Load data

print('  Loading data ...')
# training
df_train = pd.read_csv(TrainingDataFile)
X_train = df_train.iloc[2:8,1:].astype(float).T.values
y_train = df_train.iloc[45,1:].astype(float).to_frame().values  # row 62 => middle transect behind sbw

# testing
df_test = pd.read_csv(TestDataFile)
X_test = df_test.iloc[2:8,1:].astype(float).T.values
y_test = df_test.iloc[45,1:].astype(float).T.values  # row 62 => middle transect behind sbw

#%% Model training

print('  Training model (be patient!) ...')
# Build a neural network with six input variables
model = tf.keras.Sequential([
    tf.keras.layers.Dense(Nneurons_Input, activation='relu', input_shape=(6,)),  # input layer
    #tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1) # single output layer # this should change when trying to predict entire shoreline
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
fit = model.fit(X_train, y_train, epochs=Nepochs, verbose=verbose)

# loss plot
plt.figure()
plt.plot(fit.history['loss'])
plt.title('Model Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# compute model predictions - training
y_pred = model.predict(X_train, verbose=verbose)

# plot training performance
plt.figure()
plt.scatter(y_train, y_pred, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values: Training Data')

# Annotate the plot with computed metrics
metrics_df = getSkillMetrics(y_train, y_pred)
for index, row in metrics_df.iterrows():
    plt.annotate(f"{row['Metric']}: {row['Value']:.4f}", (0.05, 0.95 - 0.05 * index),
                 xycoords='axes fraction', fontsize=10)   
plt.show()

#%% Test the model on new data

print('  Applying model to test data ...')
y_pred = model.predict(X_test, verbose=verbose)

# Plot predictions vs actual values
plt.figure()
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values: Test Data')

# Annotate the plot with computed metrics
metrics_df = getSkillMetrics(y_test, y_pred)
for index, row in metrics_df.iterrows():
    plt.annotate(f"{row['Metric']}: {row['Value']:.4f}", (0.05, 0.95 - 0.05 * index),
                 xycoords='axes fraction', fontsize=10)   
plt.show()

#%%