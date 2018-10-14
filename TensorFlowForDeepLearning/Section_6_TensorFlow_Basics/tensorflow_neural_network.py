# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:42:13 2018

@author: rene
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


with tf.Session() as sess:
    #We're going to build a graph to represent our logistical regression classifier
    #in the same manner as our manual NN

    #This is to make sure our seeds are the same per instruction
    np.random.seed(101)
    tf.set_random_seed(101)

    rand_a = np.random.random(size=(5,5)) * 100
    print(rand_a)

    rand_b = np.random.random(size=(5,1)) * 100
    print(rand_b)

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    #Tf has a multiply and matmult operation
    #TF also overloaded the math operators to use these
    #operations

    add_op = a + b
    mul_op = a * b
    add_result = sess.run(add_op, feed_dict={a:10, b:20})
    print(add_result)
    #let's add our random data
    
    add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)
    
    mult_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
    print(mult_result)
    
    #Let's take another step towards building our NN

    #Our data will have 10 features
    n_features = 10
    
    #We'll have 3 neurons in our network
    n_dense_neurons = 3
    
    #Create our place holder for x
    #W is junk data of our weight
    #b is our bias 
    x = tf.placeholder(tf.float32, (None, n_features))
    W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
    b = tf.Variable(tf.ones([n_dense_neurons]))
    
    xW = tf.matmul(x,W)
    z = tf.add(xW, b)
    a = tf.sigmoid(z)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    layer_out = sess.run(a, feed_dict={x: np.random.random(size=(1, n_features))})
    print(layer_out)
    
    #SIMPLE LINEAR REGRESSION
    #Let's actually do our simple regression
    x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
    print(x_data)
    
    y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
    print(y_label)
    
    #Plot shows the linear trend
    #plt.plot(x_data, y_label, '*')
    
    #Now that we have the data, let's a model from it into an nn
    #Getting a random data point to predict from    
    #print(np.random.rand(2))
    random_data = np.random.rand(2)
    m = tf.Variable(random_data[0])
    b = tf.Variable(random_data[1])
    
    error = 0
    
    #Let's fit our training data to the model by first gathering
    #the error
    for x,y in zip(x_data, y_label):
        
        y_hat = m*x + b
        
        #Below is will be our cost function, sum of squared error
        error += (y - y_hat) ** 2
        
    #We need an optimizer to adjust off of the error
    #Let's not have a huge learning rate so as to find the global minimum correctly
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #Lets train the ml!
    training_steps = 100
    
    for i in range(training_steps):
        sess.run(train)
        
    #Now let's test the model with our random data points
    #And plot the model
    final_slope, final_intercept = sess.run([m, b])
    
    x_test = np.linspace(-1, 11, 10)
    y_pred_plot = final_slope*x_test + final_intercept
    plt.plot(x_test, y_pred_plot)
    plt.plot(x_data, y_label, '*')
    
    
    
    