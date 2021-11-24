import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.font_manager
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

XSMALL_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dataFile = "shock_interpolator/shock.pkl"
load_model = True
model_path = "problem2_model"
n_hidden_layers = 7
n_hidden_units = 50
seed = 42
learning_rate = 0.005
train_epochs = 500
mean_dist_threshold = 0.25

np.random.seed(seed)
tf.random.set_seed(seed)

def plot_loss(history, filename):
    plt.figure()
    plt.semilogy(history.history['loss'], label='Training Loss')
    plt.semilogy(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)

def reshape_pts(pts):
    pts_x = pts[:, 0::2]
    pts_y = pts[:, 1::2]
    return np.stack((pts_x, pts_y))

def correct(labels, preds, threshold):
    norm = np.linalg.norm(reshape_pts(preds) - reshape_pts(labels), axis=0)
    mean = np.mean(norm, axis=1)
    return mean < threshold

def plot_sample(features, geom_cols, labels, preds, index, filename):
    sample_mach = features.iloc[index]['mach']
    sample_geom_x = features.iloc[index][geom_cols].to_numpy()[0::2]
    sample_geom_y = features.iloc[index][geom_cols].to_numpy()[1::2]
    sample_labels_x = labels[index, 0::2]
    sample_labels_y = labels[index, 1::2]
    sample_preds_x = preds[index, 0::2]
    sample_preds_y = preds[index, 1::2]
    plt.figure()
    plt.scatter(sample_geom_x, sample_geom_y, label="Geometry")
    plt.scatter(sample_labels_x, sample_labels_y, label="Label")
    plt.scatter(sample_preds_x, sample_preds_y, label="Prediction")
    plt.axis('equal')
    plt.title("Sample Result, $M = {0:.2f}$".format(sample_mach))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=300)

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
labels_cols = []
for i in range(n_shock):
    labels_cols += ['shock_{0}_x'.format(i), 'shock_{0}_y'.format(i)]
geom_cols = []
for i in range(n_geom):
    geom_cols += ['geom_{0}_x'.format(i), 'geom_{0}_y'.format(i)]

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

# Drop NACA cases
data = data[data['type'] != 'naca0012']

# Train-test split
train_data = data.sample(frac=0.9, random_state=seed)
test_data = data.drop(train_data.index)

# Break out labels and features
train_features = train_data.copy()
test_features = test_data.copy()
train_labels = train_features[labels_cols].copy()
test_labels = test_features[labels_cols].copy()
train_features = train_features.drop(columns=labels_cols+['type'])
test_features = test_features.drop(columns=labels_cols+['type'])

if load_model:
    print("Loading model...")
    model = keras.models.load_model(model_path)
else:
    print("Building model...")

    # Build model
    model = tf.keras.Sequential([layers.BatchNormalization()])
    for i in range(n_hidden_layers):
        model.add(layers.Dense(
            units = n_hidden_units,
            kernel_initializer = 'glorot_uniform',
            bias_initializer = 'zeros',
            activation = 'relu'))
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(
        units = n_shock*2, # double for x and y coords
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        activation = 'linear'))

    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error')

    print("Training model...")

    # Train model
    history = model.fit(
        train_features,
        train_labels,
        epochs = train_epochs,
        batch_size = 32,
        verbose = 1,
        validation_split = 0.1)
    model.save(model_path)

    plot_loss(history, "loss.png")

train_preds = model.predict(np.array(train_features))
test_preds = model.predict(np.array(test_features))

train_correct = correct(train_labels.to_numpy(), train_preds, mean_dist_threshold)
test_correct = correct(test_labels.to_numpy(), test_preds, mean_dist_threshold)

print('')
print('------------------- TRAINING DATA --------------------------')
print('')
print(f'Model Correctness: {np.sum(train_correct)/train_correct.shape[0]}')
print('')
print('------------------- TEST DATA --------------------------')
print(f'Model Correctness: {np.sum(test_correct)/test_correct.shape[0]}')
print('')

correct_tests = np.argwhere(test_correct).flatten()
incorrect_tests = np.argwhere(np.logical_not(test_correct)).flatten()
plot_sample(
    test_features,
    geom_cols,
    test_labels.to_numpy(),
    test_preds,
    np.random.choice(correct_tests),
    "sample.png")

# import code; code.interact(local=locals())
