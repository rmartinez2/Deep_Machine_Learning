# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:06:47 2018

@author: rene
"""

#A tensor as an n-dimensionally array

import tensorflow as tf

hello = tf.constant("Hello ")
world = tf.constant("World")

#To be able to actually interact with the tensors, we have to create a session

#This is how you create a session and with the result we can print it out
with tf.Session() as sess:
    result = sess.run(hello+world)
    
print(result)

a = tf.constant(10)
b = tf.constant(20)

#We can build the computations for our tensors by simply scripting the operations we want to run
#Than we need to run the appended operations in a session, to demonstrate we shall print appending of 3 add operations
#If we set each of these tensor ops to a variable, we can build our functions with the shapes of our data sets
print(a + b)
print(a + b)
print(a + b)

with tf.Session() as sess:
    result = sess.run(a + b)
    
print(result)

#Some other functions that can build matrices
const = tf.constant(10)
fill_mat = tf.fill((4,4), 10)
my_zeros = tf.zeros((4,4))
my_ones = tf.ones((4,4))
my_randn = tf.random_normal((4,4), mean=0, stddev=1.0)
my_randu = tf.random_uniform((4,4), minval=0, maxval=1)


#sess = tf.InteractiveSession() creates an interactive session in tf. It is most useful in a note book or console setting
#my_ops = [const, fill_mat, my_zeros, my_ones, my_randn, my_randu]
#for ops in my_ops:
#    session.run(ops)

a = tf.constant([[1, 2],
                 [3, 4]])
print(a.get_shape())

b = tf.constant([[10], [100]])
print(b.get_shape())

matmul = tf.matmul(a,b)

with tf.Session() as sess:
    result = sess.run(matmul)
    
    
print(result)