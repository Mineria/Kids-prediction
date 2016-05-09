import pandas as pd
import numpy as np

filename = "data/allData.csv"

def read_training(filename):
    df = pd.read_csv(filename, header=None)
    return df

def prepare(dataframe):
    """
    Given a panda's dataframe. Normalize it.
    Returns a numpy matrix
    """
    df = dataframe

    def race(x):
        if (x == "black"):
            return 1
        else:
            return 0

    df['sex'] = df['sex'].map({ 'Female': 0, 'Male': 1 }).astype(int)
    df['race'] = df['race'].map( lambda x: race(x) ).astype(int)

    return df.values

def train(clf):
    """
    Given a sklearn classifier.
    Returns the trained matrix
    """
    pass

dataframe = read_training(filename)
data_matrix = prepare(dataframe)
