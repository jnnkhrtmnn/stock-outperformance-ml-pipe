# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:24:13 2021

@author: janni
"""

import numpy as np
from pathlib import Path
from joblib import load
from flask import Flask, request, jsonify



# initialisepath

#path = Path('C:/Users/janni/Desktop/blueprint/ml-blueprint-arch')

# load model
reg = load(Path('models') / 'reg.joblib') 
    
    
    

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def ml_inference():
    '''
    Uses JSON input from post method, 
    returns predicition and input as JSON

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    json_ = request.json

    X = np.array([[json_['x1'], json_['x2']]])
    
    # Some checks
    assert X.shape == (1,2)
    
    pred = reg.predict(X)
    
    return jsonify({'prediction': str(pred[0])
                    ,'input': {'x1' : str(X[0][0])
                              ,'x2' : str(X[0][1])}
                    })
 

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
