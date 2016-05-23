# --*-- coding: utf8 --*--
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np

from normalization import normalize_operands, normalize_operator

# Export the model
from sklearn.externals import joblib

filename = "data/train.csv"
algorithm_name = "k_means" # linear_regression

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

    def sum_complexity(row):
        operands = []
        operands.append(row['op1'])
        operands.append(row['op2'])
        operands.append(row['operator'])
        return sum(operands)

    def normalize_time(time):
        if time < 10:
            return 2
        if time < 20:
            return 1
        else:
            return 0

    df['time'] = df['time'].map( lambda x: normalize_time(x) ).astype(int)
    df['op1'] = df['op1'].map( lambda x: normalize_operands(x) ).astype(int)
    df['op2'] = df['op2'].map( lambda x: normalize_operands(x) ).astype(int)
    df['operator'] = df['operator'].map( lambda x: normalize_operator(x) ).astype(int)

    complexity = df.apply(sum_complexity, axis=1).astype(int)
    df.insert(4, 'complexity', complexity) # Insert column into index 3

    df = df.drop(deleted_labels, axis=1)

    return df.values

def load_classifier(algorithm_name, data, target):

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data, target)

    return clf

def prediction_cross_validation(clf, data, target):

    cross_validation_scores = cross_validation.cross_val_score(
        clf, data, target, cv=10)

    score = cross_validation_scores.mean()
    return score

def prediction_normal(clf, data, target):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=0.18, random_state=1)

    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def main(algorithm_name, filename):

    dataframe = read_training(filename)
    train_data = prepare_data(dataframe)
    data = train_data[0::, 0:4]
    target = train_data[0::, 4]

    clf = load_classifier(algorithm_name, data, target)

    cross_score = prediction_cross_validation(clf, data, target)
    normal_score = prediction_normal(clf, data, target)

    print "cross_validation_scores %f" % (cross_score * 100)
    print "normal validation is %f" % (normal_score * 100)

    print
    print"*" * 30
    print

    print "Preciting a value"

    temp = [1, 8, 3, 12]
    temp = [8, 8, 8, 24]
    # temp = np.array(temp).reshape((len(temp), 3))
    temp = np.array(temp).reshape(-1, 4)
    # http://stackoverflow.com/questions/12575421/convert-a-1d-array-to-a-2d-array-in-numpy

    print "too predict"
    print temp

    print clf.predict(temp)

    joblib.dump(clf, 'model/filename.pkl')

main(algorithm_name, filename)
