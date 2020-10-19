
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
import keras.losses
from keras.models import Sequential
from keras.layers import Dense

def create_model(data):
    
    df = pd.read_csv(data)

    df['5d_future_close'] = df['Last'].shift(-5)
    df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
    df['5d_close_pct'] = df['Last'].pct_change(5)
    df['5d_vol_pct'] = df['Volume'].pct_change(5)

    # generate evenly spaced grid for x values 0-10
    x = np.linspace(0, 10)

    # generate randomized data for y values
    y = x + np.random.standard_normal(len(x))

    # a list of the feature names for later
    feature_names = ['5d_close_pct']  

    # Create SMA moving averages and rsi for timeperiods of 14, 30, and 50
    for n in [14, 30, 50]:

     # Create the SMA indicator and divide by 'Last'
      df['ma' + str(n)] = df['Last'].rolling(n).mean().pct_change()
      feature_names += ['ma' + str(n)]
 
    # Drop all na values
    df = df.dropna()
    feature_names += ['5d_vol_pct']
    features = df[feature_names]
    targets = df['5d_close_future_pct']

    # Create DataFrame from target column and feature columns
    feat_targ_df = df[['5d_close_future_pct'] + feature_names]

    linear_features = sm.add_constant(features)
    train_size = int(0.8 * features.shape[0])
    train_features = linear_features[:train_size]
    train_targets = targets[:train_size]
    test_features = linear_features[train_size:]
    test_targets = targets[train_size:]

    # Create the linear model and complete the least squares fit
    model = sm.OLS(train_targets, train_features)
    results = model.fit()  # fit the model

    # Make predictions from our model for train and test sets
    train_predictions = results.predict(train_features)
#   test_predictions = results.predict(test_features)

    # Standardize the train and test features
    scaled_train_features = scale(train_features)
    scaled_test_features = scale(test_features)

    epochs = [200]
    layers = [[25,30,1]]

    # Create the model
    def model_func(layer):
        model_1 = Sequential()
        model_1.add(Dense(layer[0], 
            input_dim=scaled_train_features.shape[1], activation='relu'))
        model_1.add(Dense(layer[1], activation='relu'))
        model_1.add(Dense(layer[2], activation='linear'))
        return model_1

    # Fit the model
    model = 1
    max = -100000

    for epoch in epochs:
        for layer in layers:
            model_1 = model_func(layer)
            model_1.compile(optimizer='adam', loss='mse')
            history = model_1.fit(scaled_train_features, train_targets, epochs=epoch)
            train_preds = model_1.predict(scaled_train_features)
            if(max < r2_score(train_targets, train_preds)):
                max = r2_score(train_targets, train_preds)


# Use the last loss as the title



    return model_1

def test_predictions(model,data):

    df = pd.read_csv(data)


    df['5d_close_pct'] = df['Last'].pct_change(5)
    df['5d_vol_pct'] = df['Volume'].pct_change(5)

    # a list of the feature names for later
    feature_names = ['5d_close_pct']  

    # Create SMA moving averages and rsi for timeperiods of 14, 30, and 50
    for n in [14, 30, 50]:

     # Create the SMA indicator and divide by 'Last'
      df['ma' + str(n)] = df['Last'].rolling(n).mean().pct_change()
      feature_names += ['ma' + str(n)]

    feature_names += ['5d_vol_pct']
    # Drop all na values
    df = df.dropna()
    features = df[feature_names]
    test_features = sm.add_constant(features)
    scaled_test_features = scale(test_features)
    test_preds = model.predict(scaled_test_features)
    
    return test_preds, df['Last']