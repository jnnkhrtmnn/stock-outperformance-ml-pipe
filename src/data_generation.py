# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:45:10 2021

@author: janni
"""

from scipy.stats import norm
import pandas as pd
from pathlib import Path

path = Path('C:/Users/janni/Desktop/blueprint/ml-blueprint-arch')

def generate_data():
    '''
    Generates random data set with y, x1, x2 and epsilon.
    y is a linear combination of iid gaussian x1 and x2
    plus the gaussian error term epsilon

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    n= 1000
    
    x1 = norm.rvs(10,3,n)
    x2 = norm.rvs(30,5,n)
    epsilon = norm.rvs(0,1,n)
    
    y = x1 + x2 + epsilon
    
    df = pd.DataFrame(list(zip(y,x1,x2)), columns=['y', 'x1', 'x2'])
    
    df.to_pickle(path / 'data' / 'dat.pkl')
    
    print('Data generated!')
    return df



if __name__=='__main__':
    generate_data()
