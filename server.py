# --*-- coding:utf-8 --*--
from flask import Flask
app = Flask(__name__)

from sklearn.externals import joblib
from normalization import normalize_operands, operator_number_to_string
import numpy as np

operator_model = {
    "+": 'model/sum/model.pkl',
    "-": 'model/sub/model.pkl',
    "*": 'model/div/model.pkl',
    "/": 'model/mul/model.pkl'
}

@app.route("/")
def hello():
    return "Go to /predict"

@app.route("/predict/<int:op1>/<int:operator_number>/<int:op2>/<float:time>")
def predict(op1, operator_number, op2, time):

    # operator number
    # 1= sum | 2= Resta
    # 3= Multiplication | 4 = Division

    op1 = normalize_operands(op1)
    op2 = normalize_operands(op2)
    operator = operator_number_to_string(operator_number)

    filename_model = operator_model[operator]
    print filename_model
    clf = joblib.load(filename_model)

    data_input = [op1, operator, op2]
    data_input = np.array(data_input).reshape(-1, len(data_input))

    return str(clf.predict(data_input)[0])

if __name__ == "__main__":
    app.run()
