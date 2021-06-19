# -*- coding: utf-8 -*-
"""

@author: jannik
"""

#%%

import os
import json
import logging
from pathlib import Path
from joblib import load, dump
#from flask import Flask, request, jsonify

import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt
import datetime

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


#%% get data

def get_yahoo_data(tick: str):
    '''
    Uses yahoo finance API to get data of a specified security
    input: ticker name

    output: ticker data
    '''

    ticker = yf.Ticker(tick)
    ticker_data = ticker.history(period="max")

    return ticker_data


# Task: Weekly outperformance
def get_weekly_performance(dat: pd.DataFrame, col_keys: list):
    '''
    calculates weekly performance
    inputs:
        df: dataframe
        col_key: keys of columns for whih to calc. this

    output:
        pandas df of weekly returns

    '''

    f = dat.groupby([pd.Grouper(level='Date', freq='W-MON')])[col_keys].first()
    l = dat.groupby([pd.Grouper(level='Date', freq='W-MON')])[col_keys].last()

    ret = (l - f) / f

    return ret 







#%% feature engineering

def feature_engineering(wk_dat: pd.DataFrame, ticker: str, path):
    '''
    Engineers features.
    inputs:
        wk_dat: pandas df of week-based data
        ticker: ticker name
        path: project path for storing data
    output: pandas df week-based data with engineered features
    Also stored as pkl file
    '''

    # momentum, value last week
    wk_dat['perf_diff_shift_1'] =  wk_dat['perf_diff'].shift(-1)

    # mov avg over 8 weeks
    wk_dat['perf_diff_ma_8'] = wk_dat['perf_diff'].shift(-1).rolling(window=8,
                                                        min_periods=4).mean()

    wk_dat['perf_diff_ma_4'] = wk_dat['perf_diff'].shift(-1).rolling(window=4,
                                                        min_periods=2).mean()


    wk_dat['perf_diff_std_4'] = wk_dat['perf_diff'].shift(-1).rolling(window=4,
                                                        min_periods=4).std()


    wk_dat['stock_ma_8'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=8,
                                                        min_periods=4).mean()

    wk_dat['stock_ma_4'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=4,
                                                        min_periods=2).mean()


    wk_dat['stock_std'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=4,
                                                        min_periods=4).std()

    wk_dat['index_std'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=4,
                                                        min_periods=4).std()


    wk_dat.dropna(inplace=True)
    wk_dat.reset_index(inplace=True)

    wk_dat.to_pickle(path / "wk_dat.pkl")

    return wk_dat




#%% Putting it all together

def load_data_and_engineer_features(root_path):
    '''
    Function for all loading and feature engineering.
    input:
        root path of project, where config is located
    output:
        pd DataFrame of weekly data of features / target
    '''
    path = Path(root_path)

    with open(path / 'config.json') as f:
        config = json.load(f)


    ticker = config['stock_ticker']
    bm_ind = config['benchmark_index_ticker']

    # desired outperformance in percentage points
    outp_thresh = config['outperformance_threshold']
    test_size = config['test_split_size']


    benchmark_data = get_yahoo_data(tick=bm_ind)
    ticker_data = get_yahoo_data(tick=ticker)

    ticker_data[ticker+'_median_price'] = ticker_data[['High', 'Low']].median(axis=1)
    benchmark_data[bm_ind+'_median_price'] = benchmark_data[['High', 'Low']].median(axis=1)

    # join data
    dat = ticker_data[[ticker+'_median_price']].join(benchmark_data[[bm_ind+'_median_price']],
                                            how="outer")

    # drop where NA/NaN
    dat.dropna(inplace=True)

    wk_dat = get_weekly_performance(dat=dat, col_keys=[ticker+'_median_price', bm_ind+'_median_price'])

    # construct target

    wk_dat['perf_diff'] = wk_dat[ticker+'_median_price'] - wk_dat[bm_ind+'_median_price']
    wk_dat['target'] = wk_dat['perf_diff'] >= outp_thresh

    wk_dat = feature_engineering(wk_dat=wk_dat, ticker=ticker, path=path)

    return wk_dat



#%% Further functions for feature handling

x_lst =['perf_diff_shift_1'
       ,'stock_std'
       ,'index_std'
       ,'perf_diff_ma_8'
       ,'perf_diff_ma_4'
       ,'perf_diff_std_4'
       ,'stock_ma_8'
       ,'stock_ma_4']

def feature_select(wk_dat: pd.DataFrame, list_of_features: list):
    '''
    Selects features. Here based on a list, but of course other methods could go here
    (or in any other pipeline, eg sklearn)
    input: 
        wk_dat: pandas df of week-based data
        list_of_features: list of features
    output: X and y, as in any data science project
    '''

    X = wk_dat[list_of_features].values
    y = wk_dat['target'].values

    return X, y

#%%

def ts_train_test_split(X, y, test_size):
    '''
    Custom traint est split for time series
    inputs:
        X,y: Arrays of X and y
        test_size: float of test size fraction
    outputs:
        X_train, X_test, y_train, y_test, y_test arrays
    '''
    train_index = range(0,int(X.shape[0]*(1-test_size)))
    test_index = range(int(X.shape[0]*(1-test_size)),X.shape[0])

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test


#%%
if __name__ == '__main__':
    dir = os.path.dirname(__file__)
    path = Path(dir[:-3])
    load_data_and_engineer_features(root_path=path)



# %%
