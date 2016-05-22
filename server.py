# --*-- coding:utf-8 --*--
from flask import Flask
app = Flask(__name__)

from sklearn.externals import joblib
from normalization import normalize_operands, server_operation_conversion

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

    data_to_predict = [op1, operator, op2, complexity]

    print data_to_predict

    return str(clf.predict(data_to_predict))


  # Aquí va la magia de Machile Learning
  # return float(regr.predict([v])[0])

if __name__ == "__main__":
    app.run()

# Desde Rails podéis llamarlo así

# def estimate
#   self.estimation = HTTP.get("http://localhost:5000/predict/#{op1.to_f}/").to_s
# end
