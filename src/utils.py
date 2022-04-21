import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd


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


def fancy_legend(leg):
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(1)


def autocorrelogram(df: pd.DataFrame, symbol: str, link: str,
                    partial: bool = False, squared: bool = False, 
                    column: str = 'y_plr',
                    lag_method: str = 'Hyndman') -> None:
    """Autocorrelogram/Partial Autocorrelogram plot of a given variable
       inside column.
    Args:
        df (pd.DataFrame): data,
        symbol (str): Stock index, 
        link (str): link at with the data are taken,
        partial (bool, optional): to choose between the autocorrelogram
                                  and the partial autocorrelogram.
                                  Defaults to False,
        column (str, optional): df's column to plot. Defaults to 'y_plr',
        lag_method (str, optional): method to compute the maxlag.
                                    Defaults to 'Hyndman'.
    """
    z = df[column].to_numpy()
    n = len(z)
    
    if lag_method == 'Default':
        maxlag = np.ceil(10. * np.log10(n))
    elif lag_method == 'Box-Jenkins':
        maxlag = np.ceil(np.sqrt(n) + 45)
    elif lag_method == 'Hyndman':
        maxlag = np.min((n / 4., 10.))

    string = ' '
    if partial:
        kind = 'Partial Autocorrelogram'
        ylabel = 'Pacf value'
        Aut_Fun_z = pacf(z, nlags=maxlag)[1:]
        start = 1
        yticks = np.arange(-0.1, 0.1, 0.1)
    else:
        kind = 'Autocorrelogram'
        ylabel = 'Acf value'

        if squared:
            z = df[column].to_numpy() ** 2.
            string = ' Squared'            

        # fft = False to avoid warning
        Aut_Fun_z = acf(z, nlags=maxlag, fft=False)
        start = 0
        yticks = np.arange(0, 1.25, 0.25)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid()
    for i, y in enumerate(Aut_Fun_z, start=start):
        if y > 0:
            plt.vlines(x=i, ymax=y, ymin=0, colors='k')
        elif y < 0:
            plt.vlines(x=i, ymax=0, ymin=y, colors='k')
    
    for ci, color in zip(['0.90', '0.95', '0.99'], ['r', 'b', 'g']):
        CI = norm.ppf((1. + float(ci)) / 2.) / np.sqrt(n)
        text = f"ci {int(float(ci) * 100.)}%"  
        plt.plot([-1, maxlag+1], [CI]*2, color +'.-.', alpha=0.6, label=text)
        plt.plot([-1, maxlag+1], [-CI]*2, color +'.-.', alpha=0.6)

    first_day = df.loc[0, 'Date']
    last_day = df.loc[n - 1, 'Date']
    #fig.subplots_adjust(bottom=0.25)
    leg = plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fancy_legend(leg)
    plt.xlabel('Lag')
    plt.ylabel(ylabel)   
    plt.suptitle("University of Roma \"Tor Vergata\" - Corso di Metodi"
               + " Probabilistici e Statistici per i Mercati Finanziari \n"
               + f" {kind} of S&P 500{string}Percentage Logarithm Returns"
               + f" from {first_day} to {last_day}")
    plt.title(f"Path length {n} sample points. Data from Yahoo Finance " 
              + f"{symbol} - {link}")
    ax.set_xticks(range(start, int(maxlag) + 1))
    ax.set_yticks(yticks)
    plt.xlim((-0.5 + start, maxlag + 0.5))
