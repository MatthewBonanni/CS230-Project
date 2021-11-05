import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Set the font and formating for figure plotting
font = {'family' : 'serif',
		'serif'  : 'Times',
        'size'   : 14}
plt.rc('font', **font)
rc('text',usetex=True)

# Read in data file
dataFile = "TBM.csv"
data_raw = pd.read_csv(dataFile, sep=',', index_col=0)


# Read in keras models
model_linear = keras.models.load_model('problem1_linearmodel')
model_deep = keras.models.load_model('problem1_deepmodel')

data = data_raw.loc[data_raw['strong'] == False]
data = data.dropna(how='any')
data = data.drop('strong', axis=1)

data_m1p1_ = data[data["M"]==2.0306122448979598]
# data_m1p1_ = data[data["M"] == 1.5]
#data_m1p1_ = data[data["M"] == 8.0]

# import code; code.interact(local=locals())
data_m1p1 = data_m1p1_[data_m1p1_["gamma"]==1.1]

data_m1p1_features = data_m1p1.copy()
data_m1p1_labels = data_m1p1_features.pop('beta')

linear_pred = model_linear.predict( np.array( data_m1p1_features  )  )
deep_pred = model_deep.predict( np.array( data_m1p1_features  )  )

fig, ax = plt.subplots()

ax.plot(data_m1p1_features["theta"], data_m1p1_labels.values, marker='o', color='black', label='Analytic Solution')
ax.plot(data_m1p1_features["theta"], linear_pred, marker='o', color='r', label='Linear Model')
ax.plot(data_m1p1_features["theta"], deep_pred, marker='o', color='b', label='Deep Model')

ax.set_xlabel("$\\theta$")
ax.set_ylabel("$\\beta$")
ax.set_title("$\\theta$, $\\beta$, plot for $M=2.03$ with $\\gamma=1.1$")
ax.legend()


deep_correct = np.count_nonzero(np.abs(deep_pred.flatten() - data_m1p1_labels.values) < 3.0)

# import code; code.interact(local=locals())
plt.show()


