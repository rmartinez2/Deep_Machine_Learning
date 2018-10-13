# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:49:43 2018

@author: rene
"""

import manual_neural_network
from manual_neural_network import Graph, Variable, PlaceHolder
from manual_neural_network import multiply, add, matmul, Session

g = Graph()
g.set_as_default()

A = Variable(10)
b = Variable(1)
x = PlaceHolder()

y = multiply(A, x)
z = add(y, b)


sess = Session()
result = sess.run(operation=z, feed_dict={x:10})
