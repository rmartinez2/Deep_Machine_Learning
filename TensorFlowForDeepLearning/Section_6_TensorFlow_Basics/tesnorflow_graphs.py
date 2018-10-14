# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:18:42 2018

@author: rene
"""

#Graphs are sets of connected nodes
#Edges are the connection
#In TF each node is an operation that provides an output


import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
    
print(result)

#Our default graph
print(tf.get_default_graph())

#A new graph
g = tf.Graph()

print(g)

#How do we set a new default graph
graph_one = tf.get_default_graph()

graph_two = g

#This is how we set it, but will only remain default
#in the with code block

#We only use this if we want to reset our graph
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())
