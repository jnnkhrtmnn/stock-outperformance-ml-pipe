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



##Ideas


Any index, so maybe DAX; does not matter really
weekly prob scores of outperformance -> binary classification, outperfom by x percent or not

A particular stock: to be picked beforehand, not neccesserilay methodic itself


then inferenceservice for use as trading strategy
add add checks for assumptions, concept drift etc

no twds "How I use an LSTM for stock picking" article, using past daily stock prices

model tuning and hpyerparameter search: consider is solved task, OOS

serialize models, use mlflow

simple sklearn models, k fr this assessment, in production use may be different due to limited retraining abilities


Transaction costs, i.e. fees, market moving effect, and taxes are not considered

# Assumed for next week, 


f1 score with overweight prec, as default action: buy index, less risky, and precision above 50% neccessaryto get outperf in real world

brier score loss, not good for imb data, otherwise liked very much as it uses probas (more information)


# use a couple of simple, rather technical features
#  understand, that you do not really care
# and my spare time is on a budget at the moment ;)
