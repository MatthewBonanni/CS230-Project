import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import seed
seed(1)
tf.random.set_seed(2)
dataFile = "TBM.csv"
seed = 42
learning_rate = 0.001

beta_threshold = 3.0

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
    epochs = 100,
    verbose = 1,
    validation_split = 0.1)

plot_loss(history_linear, "loss_linear.png")

# Build and train deep model
print("\nDeep model...\n")

model_deep = tf.keras.Sequential([
   layers.BatchNormalization(),
   layers.Dense(
       units = 20,
       kernel_initializer = 'glorot_uniform',
       bias_initializer = 'zeros',
       activation = 'relu'),
   layers.BatchNormalization(),
   layers.Dense(
       units = 20,
       kernel_initializer = 'glorot_uniform',
       bias_initializer = 'zeros',
       activation = 'relu'),
   layers.BatchNormalization(),
   layers.Dense(
       units = 20,
       kernel_initializer = 'glorot_uniform',
       bias_initializer = 'zeros',
       activation = 'relu'),
   layers.BatchNormalization(),
   layers.Dense(
       units = 20,
       kernel_initializer = 'glorot_uniform',
       bias_initializer = 'zeros',
       activation = 'relu'),
   layers.BatchNormalization(),
   layers.Dense(
       units = 20,
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


linear_pred_train = model_linear.predict( np.array( train_features )  )
deep_pred_train = model_deep.predict( np.array( train_features )  )

linear_pred_test = model_linear.predict( np.array( test_features )  )
deep_pred_test = model_deep.predict( np.array( test_features )  )

lin_train_correct = np.count_nonzero(np.abs(linear_pred_train.flatten() - train_labels.values) < beta_threshold)
deep_train_correct = np.count_nonzero(np.abs(deep_pred_train.flatten() - train_labels.values) < beta_threshold)

lin_test_correct = np.count_nonzero(np.abs(linear_pred_test.flatten() - test_labels.values) < beta_threshold)
deep_test_correct = np.count_nonzero(np.abs(deep_pred_test.flatten() - test_labels.values) < beta_threshold)

print('')
print('------------------- TRAINING DATA --------------------------')
print('')
print(f'Linear Model Correctness: {lin_train_correct/train_labels.values.shape[0]}')
print(f'Deep Model Correctness: {deep_train_correct/train_labels.values.shape[0]}')
print('')
print('------------------- TEST DATA --------------------------')
print(f'Linear Model Correctness: {lin_test_correct/test_labels.values.shape[0]}')
print(f'Deep Model Correctness: {deep_test_correct/test_labels.values.shape[0]}')
print('')

model_linear.save('problem1_linearmodel')
model_deep.save('problem1_deepmodel')
# import code; code.interact(local=locals())
