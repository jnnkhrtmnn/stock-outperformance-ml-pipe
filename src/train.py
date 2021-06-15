# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:06:15 2021

@author: janni
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump


path = Path('C:/Users/janni/Desktop/blueprint/ml-blueprint-arch')

# load data
df = pd.read_pickle(path / 'data' / 'dat.pkl')


def train_model(df: pd.DataFrame):
    '''
    trains lin reg model.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    reg : TYPE
        DESCRIPTION.

    '''
    
    x_cols =['x1', 'x2']

    X = df[x_cols]
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        ,test_size=0.30
                                                        ,random_state=42)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    
    print('Training done!')
    
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print('Test MSE: {} on {} test samples'.format(mse, len(y_pred)))
    
    dump(reg, path / 'models' / 'reg.joblib') 
    print('Model saved at: {}'.format(path / 'models' / 'reg.joblib'))
    
    return reg
    
    


if __name__=='__main__':
    train_model(df)
