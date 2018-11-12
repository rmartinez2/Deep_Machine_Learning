# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:05:01 2018

@author: rene
"""

#Use a DNNRegressor and use 6 nodes in the hidden layer

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_set = pd.read_csv('cal_housing_clean.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.30, random_state = 0)

min_max_scalar_x = MinMaxScaler()
X_train = min_max_scalar_x.fit_transform(X_train)
X_test = min_max_scalar_x.transform(X_test)

#sc = StandardScaler()
#y_train = y_train.reshape(y_train.shape[0], 1)
#y_train = sc.fit_transform(y_train)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [housingMedianAge, totalRooms, population, households, medianIncome]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
regressor = tf.estimator.DNNRegressor(feature_columns=feat_cols, hidden_units=[6, 6, 6])
regressor.train(input_func=input_func, steps=1000)