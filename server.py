# --*-- coding:utf-8 --*--
from flask import Flask
app = Flask(__name__)

from sklearn.externals import joblib
from normalization import normalize_operands, server_operation_conversion
import numpy as np

clf = joblib.load('model/filename.pkl')

@app.route("/")
def hello():
    return "Go to /predict"

@app.route("/predict/<int:op1>/<int:operator_number>/<int:op2>/<float:time>")
def predict(op1, operator_number, op2, time):

    op1 = normalize_operands(op1)
    op2 = normalize_operands(op2)
    operator = server_operation_conversion(operator_number)
    complexity = op1 + op2 + operator

    data_input = [op1, operator, op2, complexity]
    data_input = np.array(data_input).reshape(-1, len(data_input))

    return str(clf.predict(data_input)[0])

if __name__ == "__main__":
    app.run()
