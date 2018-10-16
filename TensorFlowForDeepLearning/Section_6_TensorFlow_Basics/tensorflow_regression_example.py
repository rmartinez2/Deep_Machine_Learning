# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 08:04:09 2018

@author: rene
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


import sklearn.model_selection as ms
#Let's create a large data set 

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

#We're still using linear regression
#y = mx + b
#where b = 5

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

#Let's make one df to represent our data
my_data = pd.concat([x_df, y_df], axis=1)

print(my_data.head())

#Let's plot a tiny sample of our data
#my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')

#A million points is very large to try and train our model in full gradient descent and definitely too large to train stochastically
#Mini batch is a realistic solution

#Batch size is adjustable
batch_size = 8

randos = np.random.randn(2)

#Randomly select our variables for slope and bias
m = tf.Variable(randos[0])
b = tf.Variable(randos[1])

#Create the place holders
xph = tf.placeholder(tf.float64, [batch_size])
yph = tf.placeholder(tf.float64, [batch_size])

#Let's build our graph
y_model = m*xph + b

#Now for our cost function, same cost function of MSE
error = tf.reduce_sum(tf.square(yph - y_model))

#Let's get our optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    #This is how many epochs, can be adjustable too
    batches = 10000
    
    #Since the data is pretty linear, we can get away with a small number of epochs
    
    for i in range(batches):
        
        #Randomly pulling data points from the training set
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        
        #Build the actual feed dictionary from the random indexes that are from the training set
        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}

        sess.run(train, feed_dict=feed)
        
    model_m, model_b = sess.run([m,b])
    
    print(model_m)
    print(model_b)
    
    y_hat = x_data*model_m + model_b
    
    my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(x_data, y_hat, 'r')


#Now let's solve the same thing but using the tf.estimator API 
#We have a LinearClassifier, LinearRegressor, DNNClassifier, DNNRegressor (DNN - Densely Connected NN)

#To use this we must do the following

#Define a list of feature columns
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

#Create the Estimator Model
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


#Train Test Split with sklearn
x_train, x_eval, y_train, y_eval = ms.train_test_split(x_data, y_true, test_size=0.3, random_state=101)

#Create a Data Input Function
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

#Create Train Input Function
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)

#Create Eval Input Function
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

#Call train, evaulate, and predict methods on estimator object
estimator.train(input_fn=input_func, steps=1000)

#Let's get the training metrics
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

print('TRAINING DATA METRICS')
print(train_metrics)

print('TEST SET METRICS')
print(eval_metrics)


#Now let's predict with our model
new_points = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': new_points}, shuffle=False)
pred_list = list(estimator.predict(input_fn=input_fn_predict))
predictions = []

for pred in pred_list:
    predictions.append(pred['predictions'])
    
print(predictions)

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(new_points, predictions, 'r*')