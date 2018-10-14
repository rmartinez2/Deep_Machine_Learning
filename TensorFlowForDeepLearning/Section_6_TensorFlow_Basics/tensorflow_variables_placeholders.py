# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:24:26 2018

@author: rene
"""

import tensorflow as tf

#Refer to what you've learned about Placeholders and
#Variables, those same principles apply to tensorflow

#For place holders, they must know the data type
#and the shape of the data

#Variables must be initialized

with tf.Session() as sess:
    
    my_tensor = tf.random_uniform((4,4), 0, 1)
    print(my_tensor)
    
    my_var = tf.Variable(initial_value=my_tensor)
    
    print(my_var)
    
    #Here's how we initialized our variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #Now we can run our var
    print(sess.run(my_var))
    
    #Now let's play with placeholders
    #A common use is having one part of the matrix set to None
    #so as to fill it with data later
    ph = tf.placeholder(tf.float32, shape=(None,4))
    