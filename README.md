# Predict stock outperformance of benchmark index
Simple machine learning pipeline for building a model that predicts outperfermance of a given stock to a benchmark index.


## Approach

### Refinement of task
The aim is to predict whether a particular stock will outperform an associated market index on a weekly basis by a configurable amount of x% (I implemented this as percentage points).

These parameters (stock, index, x% outperformance) can be set in the configuration file `config.json`. 


### Data
The data is downloaded via API from yahoo finance and based on daily ticker information.
I did not put any effort into which stock or index to pick, and hence consider this out of scope for now.

For most parts I have been using the DAX `^GDAXI` performance index and DAIMLER  `DAI.DE`as stock.


### Target variable definition & feature engineering

#### Target variable
Instead of going for a regression approach, I decided to model this as a binary classification taskwith the target being outperformance by x% (`True`) or not (`False`). This is more closely aligned with a potential decision that may be made based on the models output: To overweigh this particular stock or not in the portfolio.
To calculate this in the training data, I picked the median between daily highs and lows of the stock price, assuming that this will be a price at which it is possible to buy or sell the stock, which may not be the case when only closing prices are considered. I did not consider any transaction costs such as potential fees, taxes or market moving effects. 
Returns are then calculated on a weekly basis, i.e. Monday to Friday. You could also do this as a 7-days difference, but that may make overfitting even more likely as many data points in the training data would overlap.


#### Features
I presumed a careful (pre-)selection of potential features is out of scope for now (+ my spare time is on a budget at the moment...).
The features I engineered resemble a momentum based trading strategy and hence are rolling means and standard deviationds of stock prices, index "prices", and their differences. I also added some boolean features indiciting if a stock splitor dividend payment happens during the period.

### Data Analysis
I would typically do some data analysis before building a model. As I am not expecting to build a reasonable model here, I skipped this step. Typically, i would also choose a model, or a selection of models to consider for the task based on the analysis. Here, I simply opted for a random forest.


### Modeling


data splits

I implemented a sklearn pipeline for some exemplary hyperparameter tuning on the training set and then evaluation of the performance on the test set.

At the moment, feature selection is only implemented as dropping features without any variance, but in a real project this is where I would implement it.

any model really,


model tuning and hpyerparameter search: consider is solved task, OOS

serialize models, use mlflow

simple sklearn models, k fr this assessment, in production use may be different due to limited retraining abilities

like models with predict proba, as it gives more information than just a label
-> also important for trading strategy, not assessed here but probably certainty of 
outperformance higher, if prob is 0.99 than 0.51


## Evaluation


#### validation scheme: eval metrics, measures taken against overfitting, assess overall model
![Image of Yaktocat](mlruns/comp_runs.PNG)

f1 score with overweight prec, as default action: buy index, less risky, and precision above 50% neccessaryto get outperf in real world

brier score loss, not good for imb data, otherwise liked very much as it uses probas (more information)

overfitting: most important: dont trick yourself!
proer train (dev), test split

overfitting: use same model / models tructure for similar time series, eg other DAX automotive OEMs
check best parameter choices, model stability, performance,
unlikely, that only "works" for DAIMLER

simple model -> more complex model,
stop, if no improvement
#

shuffle target, add noise


ML: non-linear conditional mean
oter TS methods better
In general, I believe that other methods, eg HMM, standard econometric modeling etc

also: for entire test set, no check, if preds actually good
so failing dramatically


## Going forward
#### model -> trading strategy

then inferenceservice for use as trading strategy
add add checks for assumptions, concept drift etc
add in future events like dividend payouts

model and data versioning

trading strategy:
cetrainty of ml decision, outperf or not
portfolio view as a whole / risk model


technical side: see blueprint


## Setup & environment
For this project I have been using a conda environment, specified as
`conda create -n example_env python=3.8.10`,\
activated with
`conda activate blueprint`,\
then `cd` to the root folder of your project and installed all requirements with 
`pip install -r requirements.txt`.
Other environement shou also work as long as the runtime is the same as well as all mentioned dependencies are installed.