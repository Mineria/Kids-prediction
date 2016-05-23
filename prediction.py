# --*-- coding: utf8 --*--
# Matrix
from normalization import normalize_operator, normalize_time, normalize_operands
from pandas import DataFrame
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
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

    df['op1'] = df['op1'].map(lambda x: normalize_operands(x)).astype(int)
    df['op2'] = df['op2'].map(lambda x: normalize_operands(x)).astype(int)
    df['operator'] = df['operator'].map(lambda x: normalize_operator(x)).astype(int)
    df['time'] = df['time'].map(lambda x: normalize_time(x)).astype(int)

    # df = df.query('operator==1')


    df = df.drop(labels_to_delete, axis=1)

    return df.values

def load_classifier():
    # You can put different classifier per operator
    clf = RandomForestClassifier(n_estimators=100)
    return clf

def fit_classifier(clf, data, target):
    print "Fiting classifier..."
    clf.fit(data, target)
    return clf

def export_model(clf):
    joblib.dump(clf, model_filename)

def main():
    train_data = load_data(filename, operation="sum")
    data = train_data[0::, 0:3]
    target = train_data[0::, 3]

    clf = load_classifier()
    clf = fit_classifier(clf, data, target)

    print data
    print target

    temp = [4, 3, 6]
    temp = np.array(temp).reshape(-1, len(temp))
    print clf.predict(temp)

    export_model(clf)

main()
