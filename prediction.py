import pandas as pd
import numpy as np

filename = "data/allData.csv"

def read_training(filename):
    df = pd.read_csv(filename, header=0)
    return df

def prepare(dataframe):
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

    df['time'] = df['time'].astype(float)
    df['operator'] = df['operator'].map( lambda x: normalize_operator(x) ).astype(int)
    df['op1'] = df['op1'].map( lambda x: normalize_operands(x) ).astype(int)
    df['op2'] = df['op2'].map( lambda x: normalize_operands(x) ).astype(int)

    df = df.drop(deleted_labels, axis=1)

    return df.values

def train(clf):
    """
    Given a sklearn classifier.
    Returns the trained matrix
    """
    pass

dataframe = read_training(filename)
data_matrix = prepare(dataframe)
