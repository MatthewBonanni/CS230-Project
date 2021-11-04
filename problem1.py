import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataFile = "TBM.csv"
seed = 42
learning_rate = 0.1

def plot_loss(history, filename):
  plt.figure()
  plt.semilogy(history.history['loss'], label='loss')
  plt.semilogy(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.savefig(filename, dpi=300)

## Read data and build train, dev, and test sets
data_raw = pd.read_csv(dataFile, sep=',', index_col=0)

data = data_raw.loc[data_raw['strong'] == False]
data = data.dropna(how='any')
data = data.drop('strong', axis=1)

train_data = data.sample(frac=0.9, random_state=seed)
test_data = data.drop(train_data.index)

train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('beta')
test_labels = test_features.pop('beta')

## Build and train linear model
print("\nLinear model...\n")

model_linear = tf.keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(
        units = 1,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'linear')])

model_linear.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    loss='mean_absolute_error')

history_linear = model_linear.fit(
    train_features,
    train_labels,
    epochs = 10,
    verbose = 1,
    validation_split = 0.1)

plot_loss(history_linear, "loss_linear.png")

## Build and train deep model
print("\nDeep model...\n")

model_deep = tf.keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(
        units = 10,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(
        units = 10,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(
        units = 10,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(
        units = 1,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'linear')])

model_deep.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    loss='mean_absolute_error')

history_deep = model_deep.fit(
    train_features,
    train_labels,
    epochs = 100,
    batch_size = 32,
    verbose = 1,
    validation_split = 0.1)

plot_loss(history_deep, "loss_deep.png")

import code; code.interact(local=locals())