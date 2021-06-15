# Deployable Machine Learning App Blueprint
Simple blueprint for machine learning projects using a prediction API run via Flask app and Docker.

## Conda envs
Firt, create an environment with
`conda create -n blueprint python=3.8`,\
activate the environment using
`conda activate blueprint`,\
then `cd` to the root folder of your project and install all requirements with 
`conda install -r requirements.txt`.

## Data genertion
In the same cd execute `python src/generate_data.py` to generate some artificial data.


## DA
There is also a `jupyter notebook` for some EDA.

## Model training
Run `python src/train.py` in the project root directory.


## Run Flask App
To run the app, simply execute `python app.py`.

## Docker
Added a dockerfile.
The image can be built by running 
`docker build . -t mlblueprintarch:v1`.
To start the container just run `docker run -p 5000:5000 mlblueprintarch:v1`.


## API tests
Test the APIs by sending the stored POST requests (either to docker app or flask app) using POSTMAN
