# -*- coding: utf-8 -*-
""" Predicting income for American workers
Authors: Jorge Ferreiro & Carlos Reyes.
"""
import pandas as pd
import numpy as np
import pylab as P
from data_normalization import *
from sklearn.ensemble import RandomForestClassifier
import csv as csv

# Creating dataframe with CSV file
data_df = pd.read_csv('Data/data.csv', header=0);
test_df = pd.read_csv('Data/test.csv', header=0) # Load the test file into a dataframe

def normalise_data(df, test_matrix):
    """Normalising non-continuous parameters (given in string)"""

    df['relationship'] = df['relationship'].map( lambda x: relationship(x) ).astype(int)
    df['race'] = df['race'].map( lambda x: race(x) ).astype(int)
    df['sex'] = df['sex'].map({
        'Female': 0,
        'Male': 1
    }).astype(int)

    # For the test matrix there is no income
    # So we don't have to normalize it
    if not test_matrix:
        df['income'] = df['income'].map( lambda x: income(x) ).astype(int)

    # Check the our dataframse is only containing numbers
    df.dtypes[df.dtypes.map(lambda x: x=='object')]

    # Delete unused columns
    # Delete first column refering to user index
    df = df.drop(df.columns[0], axis=1)
    #df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return df.values

# Get Adults IDs from Test before delete the column
ids = test_df[test_df.columns[0]]


# Normalize Data and remove ID
train_data = normalise_data(data_df, test_matrix=False)
test_data  = normalise_data(test_df, test_matrix=True)

print "train_data[0::,1::]"
print train_data[0::,1::]
print "train_data[0::,14]"
print train_data[0::,14]

print test_data.shape

print 'Training...'
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0:11000,0:13], train_data[0:11000,14])

print "Test data shape"
print train_data.shape

print "Test data shape"
print test_data.shape

# print 'Predicting...'
predition = forest.predict(test_data[0::,0:13]).astype(int)

#score = forest.score(predition[0::,1::], test_data[0::,1::])
score = forest.score(test_data[0::,0:13], predition)
score = forest.score(train_data[0::,0:13], train_data[0::,0:14])


print "Score"
print score

#for i in range(10):
#    print output[i]

# test_data_with_output = array(test_data)
# (test_rows, test_colums) = test_data.shape
# last_col = test_colums

# for i in range(test_rows):
#     test_data_with_output[i,last_col]

predictions_file = open("adults_output.csv", "wb")
open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["AdultsId","Income"])
# open_file_object.writerows(zip(ids, output))
open_file_object.writerows(zip(output))
predictions_file.close()
print 'Done.'
