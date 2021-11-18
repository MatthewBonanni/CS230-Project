import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle, Wedge, Polygon
import scipy
from scipy.interpolate import lagrange
import pickle

def points_on_line(x0, x1, n):
    '''
    Calculate (x, y) values of points linearly spaced from x0 to x1.
    '''
    x = np.linspace(x0[0], x1[0], n)
    y = np.linspace(x0[1], x1[1], n)
    return np.vstack((x, y)).T

def points_on_square(top_left, l, n):
    '''
    Calculate (x, y) values of points linearly spaced along a square.
    '''
    top = points_on_line(top_left, top_left + [l, 0], n//4 + 1)[:-1]
    right = points_on_line(top_left + [l, 0], top_left + [l, -l], n//4 + 1)[:-1]
    bottom = points_on_line(top_left + [l, -l], top_left + [0, -l], n//4 + 1)[:-1]
    left = points_on_line(top_left + [0, -l], top_left, n//4 + 1)[:-1]
    return np.concatenate([top, right, bottom, left])

def points_on_diamond(tip, l, n):
    '''
    Calculate (x, y) values of points linearly spaced along a diamond.
    '''
    # First make a square
    square = points_on_square(tip, l / np.sqrt(2), n)
    # Take the square and rotate it
    theta = np.pi/4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
        np.cos(theta)]])
    return (rotation @ square.T).T

def points_on_wedge(tip, l, n):
    '''
    Calculate (x, y) values of points linearly spaced along a wedge.
    '''
    top = points_on_line(tip, tip + [l, l/2], n//3 + 1)[:-1]
    right = points_on_line(tip + [l, l/2], tip + [l, -l/2], n//3 + 1)[:-1]
    bottom = points_on_line(tip + [l, -l/2], tip, n//3 + 1)[:-1]
    return np.concatenate([top, right, bottom])

def points_on_triangle(tip, l, n):
    '''
    Calculate (x, y) values of points linearly spaced along a triangle.
    '''
    # First make a wedge
    wedge = points_on_wedge(np.array([0, 0]), l, n)
    # Take the wedge  and rotate it
    theta = np.pi
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
        np.cos(theta)]])
    flipped = (rotation @ wedge.T).T + tip
    return np.roll(flipped, n//2, axis=0)

def points_on_cylinder(tip, d, n):
    '''
    Calculate (x, y) values of points linearly spaced along a cylinder.
    '''
    theta = np.linspace(0, 2*np.pi, n+1)[:-1]
    x = -np.cos(theta)
    y = np.sin(theta)
    return (d/2) * np.vstack([x, y]).T

# Load data
with open('data.pkl', 'rb') as outfile:
    data = pickle.load(outfile)

# Geometries
geoms = {}
geoms['naca0012'] = np.empty((10, 2))
geoms['square'] = points_on_square(np.array([0, .5]), 1, 12)
geoms['diamond'] = points_on_diamond(np.array([0, 0]), 1, 12)
geoms['wedge'] = points_on_wedge(np.array([0, 0]), 1, 12)
geoms['triangle'] = points_on_triangle(np.array([1, 0]), 1, 12)
geoms['cylinder'] = points_on_cylinder(np.array([0, 0]), 1, 12)
breakpoint()

# Loop over geometries
shock = {}
for case, results in data.items():
    x_list, pressure_list = results
    shock[case] = []
    print(f'Processing the {case} cases')
    print('Working on case: ', end='', flush=True)
    # Loop over cases
    for i in range(len(x_list)):
        # TODO: Hack
        if i > 1: break
        print(f'{i}, ', end='', flush=True)
        x = x_list[i]
        pressure = pressure_list[i]
        # If the case doesn't exist, skip it and append None
        if x is None:
            shock[case].append(None)
            continue

        # -- Find shock -- #
        # Leftmost point, where the search begins
        x_start = -5
        # Search step size
        dx = .01
        # How far to go before quitting
        stop_distance = 5
        # Points in y-direction to start at
        n_points = 5
        y_starts = np.linspace(-4, 4, n_points)

        # Loop over all points in y
        shock_x = np.empty((n_points, 2))
        for y_idx, y_start in enumerate(y_starts):
            # Find the point closest to (x_start, y_start)
            start_index = np.argmin(np.linalg.norm(x - ([x_start, y_start]), axis=1))
            start_x = x[start_index]
            start_p = pressure[start_index]

            # Move from left to right, waiting for a pressure jump
            x_jump = np.empty((3, 2))
            jumps = [1.2, 1.3, 1.4]
            j = 0
            for i in range(int(stop_distance/dx)):
                index = np.argmin(np.linalg.norm(x - (start_x + [i * dx, 0]), axis=1))
                # If the pressure jumps, record it
                if pressure[index] > jumps[j]*start_p:
                    x_jump[j] = x[index]
                    j += 1
                    if j == len(jumps): break
            shock_x[y_idx, :] = np.mean(x_jump, axis=0)
        # Save data
        shock[case].append(shock_x)
    print()

# Pick case to plot
case = 'wedge'
number = 0
shock_x = shock[case][number]
x = data[case][0][number]
pressure = data[case][1][number]

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
x_geom = np.concatenate((geoms[case][:, 0], [geoms[case][0, 0]]))
y_geom = np.concatenate((geoms[case][:, 1], [geoms[case][0, 1]]))
plt.fill(x_geom, y_geom, color='white')
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
