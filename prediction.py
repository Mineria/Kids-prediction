# --*-- coding: utf8 --*--
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np
import csv     # imports the csv module

from normalization import is_multiplication, normalize_operands, normalize_operator, normalize_time

# Export the model
from sklearn.externals import joblib

filename = "data/train.csv"
algorithm_name = "linear_regression"
algorithm_name = "random_forest"

def read_training(filename):
    df = pd.read_csv(filename, header=0)
    return df

def prepare_data(dataframe):
    """
    Given a panda's dataframe. Normalize it.
    Returns a numpy matrix
    """
    df = dataframe
    deleted_labels = ['name']

    multiplication = df.apply(is_multiplication, axis=1).astype(int)
    df.insert(0, 'complexity', multiplication) # Insert column into index 3

    df['op1'] = df['op1'].map( lambda x: normalize_operands(x) ).astype(int)
    df['op2'] = df['op2'].map( lambda x: normalize_operands(x) ).astype(int)
    df['operator'] = df['operator'].map( lambda x: normalize_operator(x) ).astype(int)
    df['time'] = df['time'].map( lambda x: normalize_time(x) ).astype(int)


    df = df.drop(deleted_labels, axis=1)

    return df.values

def load_classifier(algorithm_name, data, target):

    print 'Applying Random Forest!'
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data, target)

    # if algorithm_name == "random_forest":
    #     print 'Applying Random Forest!'
    #     clf = RandomForestClassifier(n_estimators=100)
    #
    # else:
    #     clf = RandomForestClassifier(n_estimators=100)


    return clf

def prediction_cross_validation(clf, data, target):

    cross_validation_scores = cross_validation.cross_val_score(
        clf, data, target, cv=10)

    score = cross_validation_scores.mean()
    return score

def prediction_normal(clf, data, target):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=0.18, random_state=1)

    score = clf.score(X_test, y_test)
    return score

def main(algorithm_name, filename):

    dataframe = read_training(filename)
    train_data = prepare_data(dataframe)
    data = train_data[0::, 0:3]
    target = train_data[0::, 3]

    print data
    print target

    clf = load_classifier(algorithm_name, data, target)

    cross_score = prediction_cross_validation(clf, data, target)
    normal_score = prediction_normal(clf, data, target)


    print data

    print "cross_validation_scores %f" % (cross_score * 100)
    print "normal validation is %f" % (normal_score * 100)

    print
    print"*" * 30
    print

    print "Preciting a value"

    # for row in data:
    #     temp = row.reshape(-1, 3)
    #     result = clf.predict(temp)
    #     print "nop" + str(int(result[0]))
    #     if int(result[0]) == 1:
    #         print temp
    #         print "\tHooooooray"
    #     #np.array(temp).reshape(-1, 3)

    # http://stackoverflow.com/questions/12575421/convert-a-1d-array-to-a-2d-array-in-numpy

    # f = open('data/train.csv', 'rb') # opens the csv file
    # try:
    #     reader = csv.reader(f)  # creates the reader object
    #     for row in reader:   # iterates the rows of the file in orders
    #         op1 = row[1]
    #         operator = row[2]
    #         op2 = row[3]
    #
    #         temp = []
    #         temp.append(op1)
    #         temp.append(operator)
    #         temp.append(op2)
    #
    #         print row    # prints each row
    # finally:
    #     f.close()      # closing

    temp = [200, 1000, 200]
    temp = np.array(temp).reshape(-1, 3)
    print temp
    print clf.predict(temp)

    print "*****"


    temp = np.array(temp).reshape(-1, 3)
    print temp
    print clf.predict(temp)

    print "*****"

    temp = [0, 1000, 1]
    temp = np.array(temp).reshape(-1, 3)
    print temp
    print clf.predict(temp)

    print "*****"

    temp = [1, 2, 4]
    temp = np.array(temp).reshape(-1, 3)
    print temp
    print clf.predict(temp)

    print "*****"

    temp = [10, 2, 8]
    temp = np.array(temp).reshape(-1, 3)
    print temp
    print clf.predict(temp)

    joblib.dump(clf, 'model/filename.pkl')

main(algorithm_name, filename)
