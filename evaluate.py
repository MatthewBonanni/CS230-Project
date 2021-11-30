import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import lagrange

import shock_interpolator.main as shock_functions

def correct(labels, preds, threshold):
    norm = np.linalg.norm(reshape_pts(preds) - reshape_pts(labels), axis=0)
    mean = np.mean(norm, axis=1)
    return mean < threshold

# Choose case and number
case = 'square'
number = 0

# Path to trained model
model_path = "problem2_model"
# Load model
model = keras.models.load_model(model_path)

# Read data
with open('shock_interpolator/shock.pkl', 'rb') as file_name:
    shock = pickle.load(file_name)
with open('shock_interpolator/data.pkl', 'rb') as file_name:
    data = pickle.load(file_name)

# Get features and labels
features = np.concatenate([[shock[case][0][number]],
    shock[case][2][number].flatten()]).reshape(1, -1)
labels = shock[case][1][number].reshape(1, -1)

# Perform prediction and reshape
preds = model.predict(features).reshape(-1, 2)

ref_plot_points = np.linspace(-1, 1, 100)
n_points = 5
ref_nodes = np.polynomial.chebyshev.chebroots([0]*n_points + [1])
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
fig = plt.figure(figsize=(7,7))
for case, number in zip(shock.keys(), [16, 16, 16, 16, 16, 16, 3]):
    if case == 'naca0012': continue
    if case == 'ellipse':
        mach = 1.5 + .5 * number
    else:
        mach = 1.4 + .1 * number

    features = np.concatenate([[shock[case][0][number]],
        shock[case][2][number].flatten()]).reshape(1, -1)
    # Perform prediction and reshape
    preds = model.predict(features).reshape(-1, 2)
    #shock_x = shock[case][1][number]
    shock_x = preds

    x_poly = lagrange(ref_nodes, shock_x[:, 0])
    y_poly = lagrange(ref_nodes, shock_x[:, 1])

    plt.plot(x_poly(ref_plot_points), y_poly(ref_plot_points), '-', lw=3, label = f'M = {mach}, geom = {case}')
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('y (m)', fontsize=16)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.tick_params(labelsize=12)
plt.legend()
plt.savefig('geom_comparison.png', bbox_inches='tight')

# Check the correctness of the ellipse case
correctness = []
for number in range(14):
    features_ellipse = np.concatenate([[shock['ellipse'][0][number]],
        shock['ellipse'][2][number].flatten()]).reshape(1, -1)
    preds = model.predict(features_ellipse)
    labels = shock['ellipse'][1][number].reshape(1, -1)
    correctness.append(correct(labels, preds, .25))
correctness = np.array([bool(c) for c in correctness])


shock_x = shock[case][1][number]
x = data[case][0][number]
pressure = data[case][1][number]
grad = data[case][2][number]
shock_functions.plot_data(case, number, 5, (shock_x, preds), x, pressure, grad,
        shock_functions.geoms, save_fig = True, colors = ['c', '#ff7518'],
        labels = ['Expected', 'DNN Prediction'], fig_dir = './')
