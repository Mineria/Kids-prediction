# --*-- coding: utf8 --*--
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np

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

    def normalize_operator(operator):
        if operator == "/":
            return 8 # even more complexity
        elif operator == "*":
            return 7 # more complexity
        elif operator == "-":
            return 2
        elif operator == "+":
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

    def sum_complexity(row):
        operands = []
        operands.append(row['op1'])
        operands.append(row['op2'])
        operands.append(row['operator'])
        return sum(operands)

    def normalize_time(time):
        if time < 20:
            return 1
        else:
            return 0

    df['time'] = df['time'].map( lambda x: normalize_time(x) ).astype(int)
    df['op1'] = df['op1'].map( lambda x: normalize_operands(x) ).astype(int)
    df['op2'] = df['op2'].map( lambda x: normalize_operands(x) ).astype(int)
    df['operator'] = df['operator'].map( lambda x: normalize_operator(x) ).astype(int)
    df['complexity'] = df.apply(sum_complexity, axis=1)


    grouped = df.groupby(['operator'])
    print grouped.groups
    print "---" * 30
    print grouped.agg([np.sum, np.mean, np.std])
    print "---" * 30
    # print df['complexity']

    df = df.drop(deleted_labels, axis=1)

    return df.values

def load_classificator(algorithm_name):

    if algorithm_name == "random_forest":
        print 'Applying Random Forest!'
        clf = RandomForestClassifier(n_estimators=100)
    elif algorithm_name == "linear_regression":
        print 'Applying Linear Regression!'
        clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    else:
        clf = RandomForestClassifier(n_estimators=100)

    return clf

def prediction_cross_validation(clf, train_data):
    data = train_data[0::, ::2]
    target = train_data[0::, 3]

    cross_validation_scores = cross_validation.cross_val_score(
        clf, data, target, cv=10)

    score = cross_validation_scores.mean()
    return score

def prediction_normal(clf, train_data):
    data = train_data[0::, ::2]
    target = train_data[0::, 3]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=0.18, random_state=1)

    clf = clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    return score


# def train(clf, data_matrix):
#     """
#     Given a sklearn classifier.
#     Returns the trained matrix
#     """
#     clf = clf.fit(data_matrix[0::, 0:2], data_matrix[0::, 3])
#     return clf

# def prediction(model, clf, train_data, test_data, cv_iterations=5):
#     data = test_data[0::, ::2]
#     target = test_data[0::, 3]
#
#     predition = clf.predict(test_data[0::,0:2]).astype(int)
#
#     #score = forest.score(predition[0::,1::], test_data[0::,1::])
#     score = clf.score(test_data[0::,0:2], predition)
#     score = clf.score(train_data[0::,0:2], train_data[0::,0:3])
#
#     print "Score"
#     print score
#
#
#     print test_data
#     print target
#     print data
#     # clf = svm.SVC(kernel='linear', C=1)
#     scores = cross_validation.cross_val_score(clf, data, target, cv=cv_iterations)
#     return scores

def main(algorithm_name, filename):
    cv_iterations = 5

    dataframe = read_training(filename)
    train_data = prepare_data(dataframe)
    clf = load_classificator(algorithm_name)

    print train_data['complexity']

    score = prediction_cross_validation(clf, train_data)
    print "cross_validation_scores %f" % (score * 100)

    print "*" * 30

    score = prediction_normal(clf, train_data)
    print "normal validation is %f" % (score * 100)



    # clf = load_classificator(algorithm_name)
    # model = train(clf, train_data)
    #
    # scores = prediction(model, clf, train_data, test_data, cv_iterations)
    # # percentage = scores.mean() * 100
    #
    # print "Success %f" % scores
    # # print "Success %f" % percentage

main(algorithm_name, filename)
