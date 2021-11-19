import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

_XSMALL_SIZE = 12
_SMALL_SIZE = 14
_MEDIUM_SIZE = 16
_BIGGER_SIZE = 18

plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=_SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=_XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title

table = pd.read_csv("TBM_air.csv")
table = table.dropna()

im = plt.tricontourf(table['theta']*(180/np.pi), table['beta']*(180/np.pi), table['M'])
plt.colorbar(im, label=r"Mach number, $M$")
plt.xlabel(r"Wedge angle, $\theta$ (deg)")
plt.ylabel(r"Shock angle, $\beta$ (deg)")
plt.tight_layout()
plt.savefig("TBM.png", bbox_inches='tight', dpi=300)
plt.show()
