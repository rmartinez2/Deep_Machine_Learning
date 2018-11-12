# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 07:01:24 2018

@author: rene
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

#We're trying to predict the class label for any new data points, this is a binary classification example
diabetes = pd.read_csv('pima-indians-diabetes.csv')
#print(diabetes.head())

#Let's get the columns we need to normalize, this is some slight feature engineering
#done manually but should be automated
print(diabetes.columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree']

#We'll use pandas and a lambda expression to normalize our columns in this case instead of scikit learn to feature scale our data
#x_new = x-xmin / xmax - xmin
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max()-x.min()))
#diabetes.head()

#All of the data we've normalized are continuous with the exception of Age
#We will now start assigning the normalized features into feature columns using tensor flow
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#Now we have to deal with our categorical features Group and Class
#There are two different data structures for categorical data, hash buckets and vocabulary list
#This is a vocab list
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A','B','C','D'])

#Assuming that we can't apply the names of the categories in this easy kind of way, a hash bucket is better
#This will automatically create the number of categories we've supplied and are expecting
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

#Next we will discover how to convert a continuous column into a categorical one.
#Age will be the feature we will convert and is the reason we did not scale it with the other continuous values
#First we'll visualize our data to see what we're working with
diabetes['Age'].hist(bins=20)

#Based of what we saw in the histogram, we can find out how to put the ages into different buckets
#This will create buckets for each range of ages up to the boundary
#This is organization depends heavily on your data and your knowledge of your domain so this organization may not
#be suited for other cases
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

#Stitch our feature columns together into one feature column than perform a test train split
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]

#Test Train Split
#Obtain a dataframe without our labels by dropping the label feature before making a shallow copy into a var
x_data = diabetes.drop('Class', axis=1)

#Obtain our labels
labels = diabetes['Class']

X_train, X_test, y_train, y_test = ms.train_test_split(x_data, labels, test_size=0.3, random_state=101)

#Create our input layer
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

#Use a logistical regression classifier canned model and train
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)

#Evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
print(results)

#Now predict on the test data
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
print(my_pred)

#Create a DNN classifier
#Hidden units parameter defines how the NN looks, n number of inputs is the number of hidden layers, 10 represents how many
#nodes per layer, sense it's a DNN all synapses are connected to each node
#dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)

#Unfortunately, we cannot use the same input function to train the DNN, we must translate the categorical columns
#to embedding columns before we can train the model
#Dimensions represents the dimensions of the categorical column, it's 4 because we have 4 groups, A,B,C,D
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)

#Now we must replace the categorical features with the embeded categorical features
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]

input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)

#Now lets evaluate the DNN
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = dnn_model.evaluate(eval_input_func)

#We can try and play around with the number of hidden layers and nodes within them to see if we can get more accuracy.
#Keep in mind that there is a chance of overfitting
