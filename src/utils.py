import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt


def latex_settings():
    fig, ax = plt.subplots(constrained_layout=True)  
    fig_width_pt = 390.0    # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27                # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0     # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt   # width in inches
    fig_height = fig_width * golden_mean       # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 14,
              'legend.fontsize': 9,
              'xtick.labelsize': 10, 
              'ytick.labelsize': 10, 
              'figure.figsize': fig_size,  
              'axes.axisbelow': True}

    mpl.rcParams.update(params)
    return ax
