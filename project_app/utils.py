import numpy as np
import pandas as pd
import pickle
import json
import config


class Diabetes_Prediction():
    def __init__(self,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age


    def load_model(self):
        with open(config.MODEL_FILE_PATH, "rb") as f:
            self.pred_model = pickle.load(f)

        with open(config.NORMAL_SCALER_FILE_PATH, "rb") as f:
            self.normal_model = pickle.load(f)

        with open(config.JSON_FILE_PATH, "r") as f:
            self.project_data = json.load(f)


    def get_prediction(self):
        self.load_model()

        test = np.zeros(len(self.project_data["columns"]))
        test[0] = self.Glucose
        test[1] = self.BloodPressure
        test[2] = self.SkinThickness
        test[3] = self.Insulin
        test[4] = self.BMI
        test[5] = self.DiabetesPedigreeFunction
        test[6] = self.Age

        # Normalize the test array
        arr = self.normal_model.transform([test])
        prediction = self.pred_model.predict(arr)

        if prediction[0] == 1:
            return "PERSON HAVE DIABETES"
        return "PERSON DONT HAVE DIABETES"


if __name__ == "__main__":
    Glucose = 103
    BloodPressure = 30
    SkinThickness = 38
    Insulin = 83
    BMI = 43.3
    DiabetesPedigreeFunction = 0.183
    Age = 33

    dia_ins = Diabetes_Prediction(Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    dia_ins.get_prediction()
