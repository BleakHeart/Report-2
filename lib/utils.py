import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List


def latex_settings(nrows=1, ncols=1, height_factor=1.):
    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)  
    fig_width_pt = 390.0    # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27                            # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0                 # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt               # width in inches
    fig_height = fig_width * golden_mean * height_factor   # height in inches
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


def fancy_legend(leg):
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(1)


def plot_data(df: pd.DataFrame, file: str, lw: float, title: str = None,
              peaks: np.ndarray = None, feature: str = 'Load') -> None:
    ax = latex_settings()
    ax.plot(df['Date'], df[feature], 'b', lw=lw)
    
    if isinstance(peaks, np.ndarray):    
        ax.plot(df['Date'][peaks], df[feature][peaks], 'rx')
    
    ax.grid()
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{feature} (MWh)')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title(title)
    
    filepath = '../Images/'
    if os.path.isfile(filepath + file):
        pass
    else:    
        plt.savefig(f'{filepath}{file}.png', dpi=800, transparent=True)
    
    plt.show()
    

def latex_table_generator(df, filepath, float_format=None):
    n_cols = len(df.columns)
    latex_cols = [r'\textbf{%s}' %col for col in df.columns]
    with open(filepath, 'w') as tf:
        tf.write(
            df.to_latex(
                index=False,
                float_format=float_format,
                column_format='c'*n_cols,
                header=latex_cols,
                escape=False
                )
        )


def int_from_str(string: str) -> int:
    """Exctract an integer number from a string.
    Args:
        string (str): string containing a number.
    Returns:
        int: integer number inside the string.
    """
    num = int(''.join(filter(str.isdigit, string)))
    return num


def get_result_filenames(results_path: str, freq: str) -> List[str]:
    all_files = os.listdir(results_path)
    files = [file for file in all_files if file.split('_')[1] == freq]

    files.sort(key=int_from_str, reverse=False)
    return files
