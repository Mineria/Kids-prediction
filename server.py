# --*-- coding:utf-8 --*--
from flask import Flask
app = Flask(__name__)

from sklearn.externals import joblib

filename_model = 'filename.pkl'
regr = joblib.load(filename_model)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/generate_model")
def generate_model():
    joblib.dump(clf, filename_model)

@app.route("/predict/<float:op1>/<float:op2>/<path:operator>")
def predict(op1, op2, operator):
  # Aquí va la magia de Machile Learning
  return float(regr.predict([v])[0])

if __name__ == "__main__":
    app.run()

# Desde Rails podéis llamarlo así

# def estimate
#   self.estimation = HTTP.get("http://localhost:5000/predict/#{op1.to_f}/").to_s
# end
