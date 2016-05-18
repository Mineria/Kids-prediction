import pandas as pd


def sublst(row):
    return lst[row['J1']:row['J2']]


def sum_complexity(row):
    operands = []
    operands.append(row['J1'])
    operands.append(row['J2'])
    operands.append(row['J4'])
    return sum(operands)#, ow[3])

df = pd.DataFrame({'ID':['1','2','3'], 'J1': [0,2,3], 'J2':[1,4,5], 'J4':[1,4,5]})
print df
lst = ['a','b','c','d','e','f']

df['J3'] = df.apply(sublst,axis=1)
df['J4'] = df.apply(sum_complexity,axis=1)
print df
