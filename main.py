import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy
from scipy.interpolate import lagrange
import pickle

# Load data
with open('data.pkl', 'rb') as outfile:
    x, pressure = pickle.load(outfile)

# -- Find shock using the novel USL (Unintelligent Shock Locator) method -- #

# Leftmost point, where the search begins
x_start = -15
# Search step size
dx = .1
# How far to go before quitting
stop_distance = 30
# Points in y-direction to start at
n_points = 7
y_starts = np.linspace(-4, 4, n_points)

# Loop over all points in y
shock_x = np.empty((n_points, 2))
for y_idx, y_start in enumerate(y_starts):
    # Find the point closest to (x_start, y_start)
    start_index = np.argmin(np.linalg.norm(x - ([x_start, y_start]), axis=1))
    start_x = x[start_index]
    start_p = pressure[start_index]

    # Move from left to right, waiting for a pressure jump
    for i in range(int(stop_distance/dx)):
        index = np.argmin(np.linalg.norm(x - (start_x + [i * dx, 0]), axis=1))
        # If the pressure jumps, it's a shock
        if pressure[index] > 1.4*start_p:
            shock_x[y_idx, :] = x[index]
            break

# Construct interpolant for the shock
ref_nodes = np.polynomial.chebyshev.chebroots([0]*n_points + [1])
x_poly = lagrange(ref_nodes, shock_x[:, 0])
y_poly = lagrange(ref_nodes, shock_x[:, 1])
ref_plot_points = np.linspace(-1, 1, 100)

# Plot
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
fig = plt.figure(figsize=(7,7))
# Pressure field
plt.tricontourf(x[:, 0], x[:, 1], pressure, cmap='cividis')
# Solid body
cir = plt.Circle((.5, 0), .5, color = 'w')
ax = plt.gca()
ax.add_patch(cir)
# Shock polynomial
plt.plot(x_poly(ref_plot_points), y_poly(ref_plot_points), '#ff7518', lw=3)
# Shock points
plt.plot(shock_x[:, 0], shock_x[:, 1], 'ko', ms=8)
# Grid points
#plt.plot(x[:, 0], x[:, 1], 'ko', ms=3)
# Axes
plt.xlabel('x (m)', fontsize=16)
plt.ylabel('y (m)', fontsize=16)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.tick_params(labelsize=12)
#plt.legend(loc='center left', fontsize = 20)
plt.savefig('shock.pdf', bbox_inches='tight')
plt.show()
