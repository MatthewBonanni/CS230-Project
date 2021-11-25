import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import shock_interpolator.main as shock_functions

# Choose case and number
case = 'ellipse'
number = 3

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


shock_x = shock[case][1][number]
x = data[case][0][number]
pressure = data[case][1][number]
grad = data[case][2][number]
shock_functions.plot_data(case, number, 5, (shock_x, preds), x, pressure, grad,
        shock_functions.geoms, save_fig = True, colors = ['k', '#ff7518'],
        labels = ['Expected', 'DNN Prediction'], fig_dir = './')
