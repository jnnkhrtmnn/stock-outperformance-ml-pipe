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

#%%

dir = os.path.dirname(__file__)

path = Path(dir)

with open(path / 'config.json') as f:
  config = json.load(f)


#%%
ticker = config['stock_ticker']
bm_ind = config['benchmark_index_ticker']

# desired outperformance in percentage points
outp_thresh = config['outperformance_threshold']

test_size = config['test_split_size']

save_model = config['save_model']

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



# drop where NA/NaN
dat.dropna(inplace=True)


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


# momentum, value last week
wk_dat['perf_dff_shift_1'] =  wk_dat['perf_diff'].shift(-1)

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

#%% model metrics and data, algo choice

from sklearn.metrics import accuracy_score, roc_auc_score, \
                precision_score, recall_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def eval_metrics(actual, pred):
    acc = 	accuracy_score(actual, np.array(pred) > 0.5)
    roc_auc = roc_auc_score(actual, pred)
    prec = precision_score(actual, np.array(pred) > 0.5)
    rec = recall_score(actual, np.array(pred) > 0.5)
    bsl = brier_score_loss(actual, pred)
    return acc, roc_auc, prec, rec, bsl


x_lst =['perf_dff_shift_1'
       ,'stock_std'
       ,'index_std'
       ,'perf_diff_ma_8'
       ,'perf_diff_ma_4'
       ,'perf_diff_std_4'
       ,'stock_ma_8'
       ,'stock_ma_4']

X = wk_dat[x_lst].values
y = wk_dat['target'].values



#clf = LogisticRegression(random_state=0, C=0.5)
clf = RandomForestClassifier(random_state=0)

params = {'n_estimators': [20,50,100,200]
            #,'max_features': ['auto', 'sqrt']
            #,'criterion' : ['gini', 'entropy']
            ,'max_depth': [2,3,4]
            ,'bootstrap': [True, False]
            ,'class_weight' : ['balanced'] # class underweight, model might get to conservative otherwise
            }



#%% model training
tscv = TimeSeriesSplit(n_splits=5)

train_index = range(0,int(X.shape[0]*(1-test_size)))
test_index = range(int(X.shape[0]*(1-test_size)),X.shape[0])

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


from sklearn.metrics import fbeta_score, make_scorer
fbeta_scorer = make_scorer(fbeta_score, beta=0.5)


with mlflow.start_run():

        gs_cv = RandomizedSearchCV(
            estimator=clf
            ,param_distributions=params
            ,scoring=fbeta_scorer #'roc_auc' #'precision'#
            ,cv=tscv
            ,verbose=1
            ,return_train_score=True
        )

        gs_cv.fit(X_train, y_train)


        best_params = gs_cv.best_params_

        clf.set_params(**best_params)

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
        mlflow.log_param("y_test_mean", np.mean(y_test))
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("prec", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("bss", bss)
        mlflow.log_metric("pred as outperformance", np.sum(np.array(preds) > 0.5))
        mlflow.log_metric("pred as underperformance", np.sum(np.array(preds) <= 0.5))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # log model, could also be stored with joblib or, 
        # if appropriate DB available, with mlflow model registry
        mlflow.sklearn.log_model(clf, "model")

        mlflow.end_run()


#%% stage model for inference service
# I am using joblib here, but usually prefer mlflow with proper setup


if save_model:
    save_path = path / 'models' / 'clf.joblib'
    #mlflow.sklearn.save_model(clf, path / 'models' / 'clf')
    dump(clf, save_path) 
    print('Model saved at: {}'.format(save_path))
    

#%% model validation, stability,...



#%% 
# !mlflow ui
# view it at http://localhost:5000.

#%% load model

load_path = path / 'models' / 'clf.joblib'
clf = load(load_path)

