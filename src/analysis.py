import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve, find_peaks
import pywt
import holidays
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd


def linear_fit(x, y):
    m, q = np.polyfit(x=x, y=y, deg=1)
    load_lin = m * x + q
    return load_lin


def normalize(data, data_max, data_min):
    return (data - data_min) / (data_max - data_min)


def denormalize(data_normalized, data_max, data_min):
    return data_normalized * (data_max - data_min) + data_min


def q1(x):
    return x.quantile(0.025)


def q2(x):
    return x.quantile(0.975)


def fourierExtrapolation(x, n_predict, n_harm=10):
    n = x.size

    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)

    x_notrend = x - p[0] * t - p[1]     # signal detrended 
    x_freqdom = np.fft.fft(x_notrend)   # signal in frequencies domain
    f = np.fft.fftfreq(n)               # frequencies

    indexes = list(range(n))
    indexes.sort(key= lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sign = np.zeros(t.size)

    for i in indexes[:1 + n_harm * 2]:
        amplitude = np.absolute(x_freqdom[i]) / n
        phase = np.angle(x_freqdom[i])
        restored_sign += amplitude * np.cos(2 * np.pi * f[i] * t + phase)
    
    return restored_sign + p[0] * t + p[1]


def wavelet_coeffs_plot(df, waveletname='sym4', figsize=(10, 10),
                        label_size=10, title_size=14):
    """Plot of wavelen data and coefficients approximation.

    Args:
        df ([DataFrame]): [description].
        waveletname (str): ['coif5', 'sym4', 'sym5'].

    Returns:
        [tuple[array, array]]: wavelen data and coefficients.
    """
    t = np.arange(len(df))
    data = df.Load.to_numpy().copy()

    levels = pywt.dwt_max_level(len(data), waveletname)
    data_l, c_l = pywt.dwt(data, waveletname)
    lws = np.array([0.1 if i < 3 else 0.5 for i in range(levels)])

    # Empirical evidence suggests that a good initial guess for the 
    # decomposition depth is about half of the maximum possible depth
    dec = 2.
    fig, axs = plt.subplots(nrows=levels, ncols=2, figsize=figsize,
                            constrained_layout=True)
    for i in range(levels):
        axs[i, 0].plot(data_l, 'r', lw=lws[i])
        axs[i, 0].set_xlim(t[0] // dec, t[-1] // dec)
        axs[i, 1].plot(c_l, 'g', lw=lws[i])
        axs[i, 1].set_xlim(t[0] // dec, t[-1] // dec)
        axs[i, 0].set_ylabel(f"Level {i + 1}", fontsize=label_size, rotation=90)
        dec *= 2.
        if i == 0:
            axs[i, 0].set_title("Approximation coefficients",
                                fontsize=title_size, weight='bold')        
            axs[i, 1].set_title("Detail coefficients", fontsize=title_size,
                                weight='bold')        
    plt.show()
    return data_l, c_l


def wavelet_filter(data, wavelet='sym4', threshold=0.04):
    maxlev = pywt.dwt_max_level(len(data), wavelet)

    # Decompose into wavelet components, to the level selected
    coeffs = pywt.wavedec(data, wavelet, level=maxlev)

    # Apply threshold to detail coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], mode='soft', 
                                   value=threshold*max(coeffs[i]))
    
    # Multilevel 1D Inverse Discrete Wavelet Transform
    datarec = pywt.waverec(coeffs, wavelet)
    return datarec


def deconvolution(signal, psf, window=96):
    deconv = np.zeros(len(signal))
    for i in range(len(signal) // window):
        deconv[i*window:(i+1)*window +1] = deconvolve(signal[(i*window) : (i+2)*window], psf)[0]
        deconv[(i+1)*window:(i+2)*window+1] = deconvolve(signal[(i+1)*window:(i+3)*window], psf)[0]
    return deconv


def resampling_data(df, feature = 'Load', resample_cost = 'h'):
    f = {'Hour': 'mean', 'Minutes': 'mean', feature: 'sum'}
    tmp = df.copy()
    df_day = tmp.set_index('Date').resample(resample_cost).agg(f)
    df_day.reset_index(drop=False, inplace=True)
    df_day['Day'] = df_day.Date.dt.day_of_week
    poland_holidays = holidays.CountryHoliday('Poland')
    df_day['Holiday'] = df_day.Date.apply(lambda x: x in poland_holidays)
    df_day['Day id'] = df_day.Date.dt.day_of_year
    peaks_up, _  = find_peaks(df_day[feature], distance=5000)
    peaks_down, _  = find_peaks(-df_day[feature], distance=5000)
    peaks = np.concatenate(([0], peaks_down, peaks_up, [len(df_day)-1]))
    df_day.loc[peaks, feature] = np.nan
    df_day[feature].interpolate(inplace=True)

    return df_day


class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None
        
    def ADF_Stationarity_Test(self, timeseries):

        #Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag=None, maxlag=1)
        
        self.pValue = adfTest[1]
        
        if (self.pValue < self.SignificanceLevel):
            self.isStationary = 'Yes'
        else:
            self.isStationary = 'No'
        
        dfResults = pd.Series(adfTest[0:4], index=['DF Test Statistic','P-Value','# Lags Used','# Observations Used'])
        
        #Add Critical Values
        for key, value in adfTest[4].items():
            dfResults[f'Critical Value ({key})'] = value

        df = pd.DataFrame(dfResults, columns=['Dickey-Fuller Test Results'])
        df.loc['Is the time series stationary?', :] = self.isStationary
        self.Results = df
    
    def kpss_Stationarity_Test(self, timeseries):
        kpssTest = kpss(timeseries, nlags=1)
        
        self.pValue = kpssTest[1]
        
        if (self.pValue > self.SignificanceLevel):
            self.isStationary = 'Yes'
        else:
            self.isStationary = 'No'
        
        dfResults = pd.Series(kpssTest[0:4], index=['KPSS Test Statistic','P-Value','# Lags Used','# Observations Used'])
        
        #Add Critical Values
        for key, value in kpssTest[3].items():
            dfResults[f'Critical Value ({key})'] = value

        df = pd.DataFrame(dfResults, columns=['KPSS Test Results'])
        df.loc['Is the time series stationary?', :] = self.isStationary
        self.Results = df