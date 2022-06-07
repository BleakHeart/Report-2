import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from preprocessing_analysis import q1, q2
from utils import latex_settings, fancy_legend
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as mape


def add_linear_seasonal_terms(df: pd.DataFrame, new_col: str = 'Load_denoised',
                              signal_col: str = 'Load_real_comp',
                              col: str = 'Load_lin_seasonal') -> pd.DataFrame:
    """Add seasonal and trend components to the signal after forecasting. 

    Args:
        df (pd.DataFrame): data containing the forecasting,
        new_col (str, optional): new columns' name of the forecasting.
                                 Defaults to 'Load'.
        signal_col (str, optional): columns' name of the sum of the 
                                    seasonal and trend component with 
                                    the forecasted signal.
                                    Defaults to 'Load_real_comp'.
        col (str, optional): columns' name of the seasonal + trend
                             component.
                             Defaults to 'Load_lin_seasonal'.

    Returns:
        pd.DataFrame: data forecasted + the seasonal and trend components.
    """
    df = df.add(df[col], axis=0)
    # remove column used for the summation
    del df[col]
    df.columns = df.columns.str.replace(signal_col, new_col)
    return df


def psf_forecasting(df: pd.DataFrame, cut_day: str,
                    feature: str = 'Load_real_comp',
                    lin_seasonal: str = 'Load_lin_seasonal') -> pd.DataFrame:
    """Function which implements the Pattern Sequence Forecast. 
       In first place it computes the PSF for the working days and the
       holidays then it constructs the forecasts using the knowledge of
       the future days.

    Args:
        df (pd.DataFrame): Table which contains the information of the
                           timeseries and if a day is an holiday or not;
        cut_day (str): day used to split previous Table into train and
                       test set;
        feature (str, optional): timeseries information used to perform 
                                 the algorithm.
                                 Defaults to 'Load_real_comp'.
        lin_seasonal (str, optional): Column in df to use as linear and
                                      seasonal component.
                                      Defaults to 'Load_lin_seasonal'.

    Returns:
        pd.DataFrame: forecasts
    """
    train_set = df[df.index < cut_day].copy()
    test_set = df[df.index >= cut_day].copy()
    
    n_days_ahead = len(test_set.index.day_of_year.unique())
    
    working_days = train_set[~train_set.Holiday]
    festivities = train_set[train_set.Holiday]
    
    f = {feature: ['mean', q1, q2], lin_seasonal: ['mean', q1, q2]}
    pred_working_days = working_days.groupby(['Day', 'Hour']).agg(f)
    pred_festivities = festivities.groupby(['Day id', 'Hour']).agg(f)    

    forecasting = []
    for day in pd.date_range(cut_day, periods=n_days_ahead, freq="d"):
        if any(day.day_of_year == festivities['Day id']):
            forecasting.append(
                pred_festivities.loc[day.day_of_year, feature].to_numpy()
                )
        else:
            forecasting.append(
                pred_working_days.loc[day.day_of_week, feature].to_numpy()
                )

    forecasting = np.concatenate(forecasting)
    tmp = df[[lin_seasonal, feature]].copy()
    n = len(forecasting) - len(test_set)
    #print(n)

    tmp.loc[tmp.index >= cut_day,
            ['forecast', 'q1_forecast', 'q2_forecast']] = forecasting[n:, :]

    return add_linear_seasonal_terms(tmp)


def sarimax_forecasting(df, cut_day, arima_args, seasonal_args):
    train_set = df[df.index < cut_day]
    # (p,d,q)(P,D,Q,S)
    # weekly: (1,0,9) x (2,1,0,52)
    # daily: (4,0,5) x (1,1,0,7)
    model = SARIMAX(
        train_set, order=arima_args,
        seasonal_order=seasonal_args,
        freq=train_set.index.inferred_freq
        )
    fit_model = model.fit()
    forecast = fit_model.get_prediction(
        start=cut_day,
        end=df.index[-1]
        )
    return forecast.conf_int()
    

def save_sarimax_results(forecast, original_df, model): 
    df = forecast.copy()
    freq = df.index.inferred_freq 
    
    if freq == 'W-SUN':
        n_weeks = (1 + (df.index[-1] - df.index[0]).days) // 7
        time_interval = n_weeks
        interval_type = 'weeks'     
    else:
        n_days = 1 + (df.index[-1] - df.index[0]).days
        time_interval = n_days
        interval_type = 'days'
    
    original_df = original_df[original_df.index >= df.index[0]].copy()
    
    df['mean Load'] = df.mean(axis=1)
    df['Load_lin'] = original_df.Load_lin
    # add the trend to the data
    df = df.add(df['Load_lin'], axis=0).iloc[:, :-1]
    df.columns = ['q1_forecast', 'q2_forecast', 'forecast']
    df.index.name = 'Date'

    file_path = f'../Results/{model}/'
    file = f"{model}_{freq}_{time_interval}{interval_type}.csv"
    if os.path.isfile(file_path + file):
        pass
    else:    
        df.to_csv(file_path + file, index=True)


def plot_model_forecasting(df, cut_day, model, feature = 'Load',
                           file=None, ax = None, legend = None, xlabel=None):
   
    freq = df.index.inferred_freq 
    if freq == 'W-SUN':
        weeks = 52
        n_weeks = ((df.index[-1] - cut_day).days + 1) // 7
        time_interval = n_weeks
        interval_type = 'weeks'     
    else:
        weeks = 2
        n_days = (df.index[-1] - cut_day).days + 1
        time_interval = n_days
        interval_type = 'days'
        
    date = cut_day - dt.timedelta(weeks=weeks)
    cut_mask = df.index >= cut_day
    date_mask = df.index > date
    
    if not ax:
        ax = latex_settings()
        legend = True
        xlabel = 'Date'
        file = f'{model}_{freq}_{time_interval}{interval_type}.png'
    
    df.loc[date_mask, [feature]].plot(ax=ax, color='b', lw=1, legend=legend)
    df.loc[cut_mask, [feature, 'forecast']].plot(ax=ax,
                                                 color=['g', 'orange'],
                                                 lw=1, legend=legend)
    ax.fill_between(x=df[date_mask].index,
                    y1=df[date_mask].q1_forecast,
                    y2=df[date_mask].q2_forecast,
                    color='orange', alpha=0.5, lw=1)
    ax.grid()
    ax.set_ylabel('Load (MW)')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_xlabel(xlabel)
    ax.set_title(f'Poland Electricity Load ({time_interval} {interval_type})')
    
    if legend:
        leg = plt.legend(loc="lower left", ncol=4, fancybox=True,
                         framealpha=0.5, fontsize=7.5,
                         labels=['Train set', 'Test set', 'Forecast', '95% CI'])
        fancy_legend(leg)
    else:
        pass
    
    file_path = f'../Images/{model}/'
    
    if os.path.isfile(file_path + file):
        pass
    else:    
        plt.savefig(file_path + file, dpi=800, transparent=True)


def accuracy_metric(dfs, cut_dates, feature='Load'):
    accuracy, mapes, rel_err = [], [], []
    for df in dfs:
        mask = df['forecast'].notna()
        tmp = df[mask].copy()
        
        rel_err_mean = np.mean(
            (tmp['q2_forecast'] - tmp['q1_forecast']) / tmp['forecast'])
        rel_err.append(round(rel_err_mean, 4))
        
        x = df.loc[mask, feature]
        y = df.loc[mask, 'forecast']
        mapes.append(round(mape(x, y), 4))
         
        mask_ci = (tmp['q1_forecast'] <= tmp[feature]) & \
                  (tmp[feature] <= tmp['q2_forecast'])
        accuracy.append(round(len(tmp[mask_ci]) / len(tmp), 4))
        
    tmp = pd.DataFrame(
        [cut_dates, accuracy, mapes, rel_err],
        index=['cut date', 'accuracy', 'mape', 'rel err']
        ).T
    tmp.iloc[:, 1:] = tmp.iloc[:, 1:].multiply(100.)
    return tmp
