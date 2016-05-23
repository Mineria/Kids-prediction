# --*-- coding: utf8 --*--
# Matrix
from normalization import normalize_operator, normalize_time, normalize_operands
from pandas import DataFrame
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import cross_validation

# Export the model
from sklearn.externals import joblib

filename = "data/train.csv"
model_filename = "model/filename.pkl"

operator = {
    "+": {
        "name": "sum",
        "filename": "model/sum/model.pkl"
    },
    "-": {
        "name": "substraction",
        "filename": "model/sub/model.pkl"
    },
    "/": {
        "name": "division",
        "filename": "model/sub/division.pkl"
    },
    "*": {
        "name": "multiplication",
        "filename": "model/sub/multiplication.pkl"
    }
}

def load_data(filename, operation):
    df = pd.read_csv(filename, header=0)
    labels_to_delete = ['name']

    if operation == "+":
        df.apply(only_sums, axis=1).astype(int)

    df['op1'] = df['op1'].map(lambda x: normalize_operands(x)).astype(int)
    df['op2'] = df['op2'].map(lambda x: normalize_operands(x)).astype(int)
    df['operator'] = df['operator'].map(lambda x: normalize_operator(x)).astype(int)
    df['time'] = df['time'].map(lambda x: normalize_time(x)).astype(int)

    df = df.drop(labels_to_delete, axis=1)

    return df.values

def load_classifier():
    # You can put different classifier per operator
    clf = RandomForestClassifier(n_estimators=100)
    #clf = linear_model.Lasso(alpha = 0.1)
    return clf

def fit_classifier(clf, data, target):
    print "Fiting classifier..."
    clf.fit(data, target)
    return clf

def export_model(clf, model_filename):
    joblib.dump(clf, model_filename)

def predict_model(clf, X):
    print "\t" + str(X)
    temp = np.array(X).reshape(-1, len(X))
    return clf.predict(temp)

def main():
    train_data = load_data(filename, operation="+")
    data = train_data[0::, 0:3]
    target = train_data[0::, 3]

    clf = load_classifier()
    clf = fit_classifier(clf, data, target)

    print data
    print target

    # X = [4, 3, 6]
    # temp = np.array(temp).reshape(-1, len(temp))
    # print clf.predict(temp)

    print predict_model(clf, [4, 3, 6])
    print predict_model(clf, [10, 3, 4])
    print predict_model(clf, [1, 2, 4])
    print predict_model(clf, [0, 0, 0])
    print predict_model(clf, [150, 150, 150])
    print predict_model(clf, [10, 4, 6])

    export_model(clf, model_filename)

main()
