# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:42:13 2018

@author: rene
"""

import tensorflow as tf
import numpy as np

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

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a:10, b:20})
    print(add_result)
    #let's add our random data
    
    add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)
    
    mult_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
    print(mult_result)
    