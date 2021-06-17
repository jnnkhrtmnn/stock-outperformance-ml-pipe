# -*- coding: utf-8 -*-
"""

@author: jannik
"""

#%%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
#from flask import Flask, request, jsonify

import yfinance as yf
import matplotlib.pyplot as plt
import datetime

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#%% set training params


path = Path('C:/Users/janni/Desktop/lehnerinvest')

# desired outperformance in percentage points
outp_thresh = .01

ticker = 'DAI.DE'
bm_ind = '^GDAXI'


#%% get data

def get_yahoo_data(tick):
    '''
    Uses yahoo finance API to get data of a specified security
    input: ticker name

    output: ticker data
    '''

    ticker = yf.Ticker(tick)
    ticker_data = ticker.history(period="max")

    return ticker_data

benchmark_data = get_yahoo_data(tick=bm_ind)
ticker_data = get_yahoo_data(tick=ticker)


#use median between high and low, 
# assuming it is possible to buy for this price at some point
# 
ticker_data[ticker+'_median_price'] = ticker_data[['High', 'Low']].median(axis=1)
benchmark_data[bm_ind+'_median_price'] = benchmark_data[['High', 'Low']].median(axis=1)

#%%
# join data
dat = ticker_data[[ticker+'_median_price']].join(benchmark_data[[bm_ind+'_median_price']],
                                            how="outer")



# drop whee NA/NaN
dat.dropna(inplace=True)

# get weeks as well
#dat['date'] = dat.index
#dat = dat.join(dat.index.isocalendar())
#dat.set_index(['year','week', 'day'], inplace=True)

# Task: Weekly outperformance
def get_weekly_performance(df, col_keys):
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


wk_dat = get_weekly_performance(dat, [ticker+'_median_price', bm_ind+'_median_price'])



#%% construct target

wk_dat['perf_diff'] = wk_dat[ticker+'_median_price'] - wk_dat[bm_ind+'_median_price']
wk_dat['target'] = wk_dat['perf_diff'] >= outp_thresh



#%% feature engineering

# use a couple of simple, rather technical features
#  understand, that you do not really care
# and my spare time is on a budget at the moment ;)

# momentum, value last week
wk_dat['perf_dff_shift_1'] =  wk_dat['perf_diff'].shift(-1)

# mov avg over 8 weeks
wk_dat['perf_diff_ma'] = wk_dat['perf_diff'].shift(-1).rolling(window=8,
                                                     min_periods=4).mean()

wk_dat['stock_ma'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=8,
                                                     min_periods=4).mean()

wk_dat['stock_std'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=4,
                                                     min_periods=4).std()

wk_dat['index_std'] =  wk_dat[ticker+'_median_price'].shift(-1).rolling(window=4,
                                                     min_periods=4).std()







wk_dat.dropna(inplace=True)
wk_dat.reset_index(inplace=True)

#%% model metrics and data, algo choice

from sklearn.metrics import accuracy_score, roc_auc_score, \
                precision_score, recall_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

def eval_metrics(actual, pred):
    acc = 	accuracy_score(actual, np.array(pred) > 0.5)
    roc_auc = roc_auc_score(actual, pred)
    prec = precision_score(actual, np.array(pred) > 0.5)
    rec = recall_score(actual, np.array(pred) > 0.5)
    bsl = brier_score_loss(actual, pred)
    return acc, roc_auc, prec, rec, bsl


x_lst =['stock_ma'
        ,'stock_std'
        ,'perf_diff_ma'
        ,'perf_dff_shift_1'
        ,'index_std'
        ]

X = wk_dat[x_lst].values
y = wk_dat['target'].values



#clf = LogisticRegression(random_state=0, C=0.5)
clf = RandomForestClassifier(random_state=0)

#%% model training
tscv = TimeSeriesSplit(n_splits=10)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    with mlflow.start_run():
        
        clf.fit(X_train, y_train)

        preds = clf.predict_proba(X_test)[:,1]

        (acc, roc_auc, prec, rec, bss) = eval_metrics(y_test, preds)

        print("  Accuracy: %s" % acc)
        print("  ROC AUC: %s" % roc_auc)
        print("  Precision: %s" % prec)
        print("  Recall: %s" % rec)
        print("  Brier score loss: %s" % bss)

        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("y_train_mean", np.mean(y_train))
        mlflow.log_param("y_test_mean", np.mean(y_train))
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("prec", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("bss", bss)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(clf, "model")

        mlflow.end_run()


#%% 
# !mlflow ui
# view it at http://localhost:5000.

#%%


#%% model validation, stability,...


#%% store serialized model

mlflow.sklearn.save_model(clf, path / 'models' / 'clf'+)





#%% load model

logged_model = 'file:///C:/Users/janni/Desktop/lehnerinvest/mlruns/0/f604896aa7ad4687b5b722d5dffbb967/artifacts/model'

clf = mlflow.sklearn.load_model(logged_model)

# %%
