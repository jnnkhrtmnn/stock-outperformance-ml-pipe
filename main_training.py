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

import numpy as np
import pandas as pd
import random
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



from src.data_load_and_feature_engineering import *

#%%

dir = os.path.dirname(__file__)

path = Path(dir)

with open(path / 'config.json') as f:
  config = json.load(f)


#%% config parameters 
ticker = config['stock_ticker']
bm_ind = config['benchmark_index_ticker']

# desired outperformance in percentage points
outp_thresh = config['outperformance_threshold']

test_size = config['test_split_size']

save_model = config['save_model']

#%% get data and engineer features (could also be loaded from pickle)
wk_dat = load_data_and_engineer_features(path)


#%% traint est split
x_lst =['perf_diff_shift_1',
       'perf_diff_ma_8', 'perf_diff_ma_4', 'perf_diff_std_4', 'stock_ma_8',
       'stock_ma_4', 'stock_std', 'index_ma_8', 'index_std',
       'stock_split_bool', 'dividends_bool']


X_train, X_test, y_train, y_test = ts_train_test_split(wk_dat, test_size, x_lst)


random.shuffle(X_test)
#random.shuffle(y_test)

#%% model metrics and algo choice

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, \
                         brier_score_loss, fbeta_score, make_scorer
from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

clf = Pipeline([
    ('vt', VarianceThreshold()), 
    ('rf', RandomForestClassifier())
    ])


def eval_metrics(actual, pred):
    acc = 	accuracy_score(actual, np.array(pred) > 0.5)
    roc_auc = roc_auc_score(actual, pred)
    prec = precision_score(actual, np.array(pred) > 0.5)
    rec = recall_score(actual, np.array(pred) > 0.5)
    bsl = brier_score_loss(actual, pred)
    return acc, roc_auc, prec, rec, bsl

fbeta_scorer = make_scorer(fbeta_score, beta=0.5)


params = {'vt__threshold' : [0]
            ,'rf__n_estimators': [20,50,100,200]
            ,'rf__max_features': ['auto', 'sqrt']
            ,'rf__criterion' : ['gini', 'entropy']
            ,'rf__max_depth': [2,3,4,5]
            ,'rf__bootstrap': [True, False]
            ,'rf__class_weight' : ['balanced'] # class underweight, model might get to conservative otherwise
            }

tscv = TimeSeriesSplit(n_splits=5)


#%% model training

def run_training():
    with mlflow.start_run():

        rs_cv = RandomizedSearchCV(
            estimator=clf
            ,param_distributions=params
            ,scoring=fbeta_scorer #'roc_auc' #'brier_score_loss'#
            ,cv=tscv
            ,verbose=1
            ,return_train_score=True
        )

        rs_cv.fit(X_train, y_train)


        best_params = rs_cv.best_params_

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


        #%% store model for inference service
        # I am using joblib here, but usually prefer mlflow with proper setup
        if save_model:
            save_path = path / 'models' / 'clf.joblib'
            #mlflow.sklearn.save_model(clf, path / 'models' / 'clf')
            dump(clf, save_path) 
            print('Model saved at: {}'.format(save_path))
    

        mlflow.end_run()

#%% run training
if __name__ == '__main__':
    run_training()


#%% model validation, stability,...

# To dos:

# validation scheme: eval metrics, measures taken against overfitting, assess overall model
# model -> trading strategy
# update requirements, explain setup




#%% Go to MLflow UI to compare models
# !mlflow ui
# view it at http://localhost:5000.

#%% load model, if needed

load_path = path / 'models' / 'clf.joblib'
clf = load(load_path)

#%%