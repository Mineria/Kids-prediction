# --*-- coding: utf8 --*--
from normalization import normal_operands, normal_operator, normal_time

# Matrix imports
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn import cluster, datasets
from sklearn.externals import joblib # export the model

filename = "data/train.csv"

def load_data(filename):
    df = pd.read_csv(filename, header=0)
    labels_to_delete = ['name']

    df['op1'] = df['op1'].map(lambda x: normal_operands(x)).astype(int)
    df['op2'] = df['op2'].map(lambda x: normal_operands(x)).astype(int)
    df['operator'] = df['operator'].map(lambda x: normal_operator(x)).astype(int)
    df['time'] = df['time'].map(lambda x: normal_time(x)).astype(int)

    df = df.drop(labels_to_delete, axis=1)

    return df.values

def main():

    train_data = load_data(filename)
    data = train_data[0::, 0:4]
    target = train_data[0::, 3]

    clf = cluster.KMeans(n_clusters=7)
    clf.fit(data)

    print data
    print(clf.labels_[::10])

    # temp = [4, 4, 6, 0]
    # temp = np.array(temp).reshape(-1, 4)
    # print clf.predict(temp)



main()
