from flask import Flask, render_template, request, jsonify
from project_app.utils import Diabetes_Prediction
import config
import pickle
import json


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods = ["POST"])
def predict():

    data = request.form
    Glucose = eval(data["Glucose"])
    BloodPressure = eval(data["BloodPressure"])
    SkinThickness = eval(data["SkinThickness"])
    Insulin = eval(data["Insulin"])
    BMI = eval(data["BMI"])
    DiabetesPedigreeFunction = eval(data["DiabetesPedigreeFunction"])
    Age = eval(data["Age"])

    dia_ins = Diabetes_Prediction(Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    res = dia_ins.get_prediction()

    # return jsonify({"Result" : f"Prdiction of Diabetes is :- {result}"})
    return render_template("prediction.html", result = res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = config.PORT_NUMBER, debug=True)