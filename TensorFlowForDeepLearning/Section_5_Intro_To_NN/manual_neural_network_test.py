# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:49:43 2018

@author: rene
"""

import manual_neural_network
from manual_neural_network import Graph, Variable, PlaceHolder
from manual_neural_network import multiply, add, matmul, Session, Sigmoid


g = Graph()
g.set_as_default()

A = Variable(10)
b = Variable(1)
x = PlaceHolder()

y = multiply(A, x)
z = add(y, b)


sess = Session()
result = sess.run(operation=z, feed_dict={x:10})
print(result)

#data is a tuple where features are in index 0 and labels are in index 1
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

features = data[0]
labels = data[1]

#This is what our classifier model looks like
#To view the data uncomment and run the line below
x = np.linspace(0, 11, 10)
y = -x + 5
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)

#Now, we're going to manually formulate the classifier model by using linear regression
#The formula that the instructor thought best fit the data was y = -x + 5
#We than need to set the above equation = to zero, y + x -5 = 0 where y + x ends up being a matrix of size [1,1]
#Than we apply the output of the operation, z, to the sigmoid function


w = Variable([1,1])
b = Variable(-5)
x = PlaceHolder()

z = add(matmul(w,x), b)
a = Sigmoid(z)

#We have now created a simple logistical regressor which classifies between two classes of data from our data set
result = sess.run(operation=a, feed_dict={x:[8,10]})
print(result)

result = sess.run(operation=a, feed_dict={x:[2,-5]})
print(result)

