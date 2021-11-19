import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import seed
seed(1)
tf.random.set_seed(2)
dataFile = "shock_interpolator/shock.pkl"
n_hidden_layers = 5
n_hidden_units = 30
seed = 42
learning_rate = 0.001
mean_dist_threshold = 0.01

def plot_loss(history, filename):
  plt.figure()
  plt.semilogy(history.history['loss'], label='loss')
  plt.semilogy(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.savefig(filename, dpi=300)

print("Loading data...")

# Read data
with open(dataFile, 'rb') as infile:
    data_raw = pickle.load(infile)

# Determine data sizes
n_shock = data_raw['cylinder'][1][0].shape[0]
n_geom = data_raw['cylinder'][2][0].shape[0]
n_cases = 0
for family, value in data_raw.items():
    n_cases += len(value[0])

# Initialize data table
data = pd.DataFrame(index=np.arange(n_cases))
data['type'] = 0
data['mach'] = 0
for i in range(n_shock):
    data['shock_{0}_x'.format(i)] = 0
    data['shock_{0}_y'.format(i)] = 0
for i in range(n_geom):
    data['geom_{0}_x'.format(i)] = 0
    data['geom_{0}_y'.format(i)] = 0

# Fill data table
row = 0
for family, value in data_raw.items():
    for i in range(len(value[0])):
        data.loc[row, 'type'] = family
        data.loc[row, 'mach'] = value[0][i]
        for j in range(n_shock):
            data.loc[row, 'shock_{0}_x'.format(j)] = value[1][i][j, 0]
            data.loc[row, 'shock_{0}_y'.format(j)] = value[1][i][j, 1]
        for j in range(n_geom):
            data.loc[row, 'geom_{0}_x'.format(j)] = value[2][i][j, 0]
            data.loc[row, 'geom_{0}_y'.format(j)] = value[2][i][j, 1]
        row += 1

# Train-test split
train_data = data.sample(frac=0.9, random_state=seed)
test_data = data.drop(train_data.index)

# Break out labels and features
labels_cols = []
for i in range(n_shock):
    labels_cols += ['shock_{0}_x'.format(i), 'shock_{0}_y'.format(i)]
train_features = train_data.copy()
test_features = test_data.copy()
train_labels = train_features[labels_cols].copy()
test_labels = test_features[labels_cols].copy()
train_features = train_features.drop(columns=labels_cols+['type'])
test_features = test_features.drop(columns=labels_cols+['type'])

print("Building model...")

# Define layers
norm_layer = layers.BatchNormalization()
dense_layer = layers.Dense(
    units = n_hidden_units,
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    activation = 'relu')
output_layer = layers.Dense(
    units = n_shock*2, # double for x and y coords
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    activation = 'linear')

# Build model
model = tf.keras.Sequential([norm_layer])
for i in range(n_hidden_layers):
    model.add(dense_layer)
    model.add(norm_layer)
model.add(output_layer)

# Compile model
model.compile(
   optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
   loss='mean_absolute_error')

print("Training model...")

# Train model
history = model.fit(
   train_features,
   train_labels,
   epochs = 100,
   batch_size = 32,
   verbose = 1,
   validation_split = 0.1)

plot_loss(history, "loss.png")

pred_train = model.predict(np.array(train_features))
pred_test = model.predict(np.array(test_features))

train_correct = np.count_nonzero(np.abs(pred_train.flatten() - train_labels.values) < mean_dist_threshold)
test_correct = np.count_nonzero(np.abs(pred_test.flatten() - test_labels.values) < mean_dist_threshold)

print('')
print('------------------- TRAINING DATA --------------------------')
print('')
print(f'Model Correctness: {train_correct/train_labels.values.shape[0]}')
print('')
print('------------------- TEST DATA --------------------------')
print(f'Model Correctness: {test_correct/test_labels.values.shape[0]}')
print('')

model_deep.save('problem2_model')
# import code; code.interact(local=locals())
