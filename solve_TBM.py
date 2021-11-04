import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt

n = 50
beta = np.linspace(0, np.pi/2, n)
M = np.linspace(1.1, 10, n)
gamma = np.linspace(1, 2, n)

def TBM(beta, M, gamma):
    return np.arctan(2 * (1/np.tan(beta)) *
        (M**2 * np.sin(beta)**2 - 1) / 
        (M**2 * (gamma + np.cos(2*beta)) + 2))

def dTBMdbeta(beta, M, gamma):
    return np.arctan((4 * M**2 * np.sin(2*beta) / np.tan(beta) *
        (M**2 * np.sin(beta)**2 - 1)) / 
        (M**2 * (gamma + np.cos(2*beta)) + 2)**2 -
        (2 / np.sin(beta)**2 * (M**2 * np.sin(beta)**2 - 1)) /
        (M**2 * (gamma + np.cos(2*beta)) + 2) +
        (4 * M**2 * np.cos(beta)**2) /
        (M**2 * (gamma + np.cos(2*beta)) + 2))

# Compute TBM solutions
[beta_mesh, M_mesh, gamma_mesh] = np.meshgrid(beta, M, gamma, indexing='ij')
theta_mesh = TBM(beta_mesh, M_mesh, gamma_mesh)
theta_mesh[theta_mesh < 0] = np.nan

# Determine which branch each solution lies on
beta_max = np.zeros((len(M), len(gamma)))
theta_max = np.zeros((len(M), len(gamma)))
strong = np.zeros((len(beta), len(M), len(gamma))).astype(bool)
for i in range(len(M)):
    for j in range(len(gamma)):
        # Guess based on max computed value
        i_guess = np.nanargmax(theta_mesh[:,i,j])
        x0 = beta_mesh[i_guess,i,j]

        # Compute maximum via bisection method on derivative
        f = lambda beta :  dTBMdbeta(beta, M[i], gamma[j])
        beta_max[i,j] = scipy.optimize.bisect(f, x0-0.1, x0+0.1)
        theta_max[i,j] = TBM(beta_max[i,j], M[i], gamma[j])

        # Assign strong and weak
        strong[:,i,j] = beta_mesh[:,i,j] > beta_max[i,j]

# Assemble table and write data
data = pd.DataFrame({
    "theta": theta_mesh.flatten(),
    "beta": beta_mesh.flatten(),
    "M": M_mesh.flatten(),
    "gamma": gamma_mesh.flatten(),
    "strong": strong.flatten()})
data.to_csv("TBM.csv")