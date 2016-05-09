# --*-- coding: utf8 --*--
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np

filename = "data/allData.csv"
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

    def normalize_operator(operator):
        if operator == "*":
            return 4
        elif operator == "/":
            return 3
        elif operator == "+":
            return 2
        elif operator == "-":
            return 1
        else:
            return 0

    def normalize_operands(operand):
        if operand <= 10:
            return 1
        elif operand >= 10 and operand < 20:
            return 2
        elif operand >= 20 and operand < 30:
            return 3
        elif operand >= 30 and operand < 40:
            return 4
        elif operand >= 40 and operand < 50:
            return 5
        elif operand >= 50 and operand < 60:
            return 6
        elif operand >= 60 and operand < 70:
            return 7
        elif operand >= 70 and operand < 80:
            return 8
        elif operand >= 80 and operand < 90:
            return 9
        elif operand >= 90:
            return 10

    # def normalize_time(time):
    #     return time

    # df['time'] = df['time'].map( lambda x: normalize_time(x) ).astype(int)
    df['time'] = df['time'].astype(int)
    df['op1'] = df['op1'].map( lambda x: normalize_operands(x) ).astype(int)
    df['op2'] = df['op2'].map( lambda x: normalize_operands(x) ).astype(int)
    df['operator'] = df['operator'].map( lambda x: normalize_operator(x) ).astype(int)

    df = df.drop(deleted_labels, axis=1)

    return df.values

def load_classificator(algorithm_name):
    if algorithm_name == "random_forest":
        print 'Applying Random Forest!'
        clf = RandomForestClassifier(n_estimators = 100)
    else:
        clf = RandomForestClassifier(n_estimators = 100)

    return clf

def train(clf, data_matrix):
    """
    Given a sklearn classifier.
    Returns the trained matrix
    """
    clf = clf.fit(data_matrix[0::, 0:2], data_matrix[0::, 3])
    return clf

def prediction(model, clf, test_matrix, cv_iterations=5):
    data = test_matrix[0::, 0:2]
    target = test_matrix[0::, 3]
    scores = cross_validation.cross_val_score(clf, data, target, cv=cv_iterations)
    return scores

def main(algorithm_name):

    dataframe = read_training(filename)
    data_matrix = prepare_data(dataframe)
    test_matrix = data_matrix

    clf = load_classificator(algorithm_name)
    model = train(clf, data_matrix)

    scores = prediction(model, clf, test_matrix)
    percentage = scores.mean() * 100

    print "Success %f" % percentage

main(algorithm_name)
