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
X = data_set.drop('medianHouseValue', axis=1)
y = data_set['medianHouseValue']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.30, random_state = 0)

min_max_scalar_x = MinMaxScaler()
min_max_scalar_x.fit(X_train)
X_train = pd.DataFrame(data=min_max_scalar_x.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(data=min_max_scalar_x.transform(X_test), columns=X_test.columns, index=X_test.index)
#X_train = min_max_scalar_x.fit_transform(X_train)
#X_test = min_max_scalar_x.transform(X_test)
#sc = StandardScaler()
#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)
#y_train = sc.fit_transform(y_train)
#y_test = sc.transform(y_test)

#X_train = pd.DataFrame(X_train)
#X_test = pd.DataFrame(X_test)
#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)

#feat_cols = ['housingMedianAge', 'totalRooms', 'totalBedrooms','population', 'households', 'medianIncome']

#X_train.columns = feat_cols
#X_test.columns = feat_cols

housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
totalBedRooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [housingMedianAge, totalRooms, population, households, medianIncome]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
regressor = tf.estimator.DNNRegressor(feature_columns=feat_cols, hidden_units=[6, 6, 6])
regressor.train(input_fn=input_func, steps=20000)

prediction_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
pred_gen = regressor.predict(prediction_input_func)
predictions = list(pred_gen)

final_preds = []

for pred in predictions:
    final_preds.append(pred['predictions'])
    
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, final_preds)**0.5

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = regressor.evaluate(eval_input_func)
print(results)