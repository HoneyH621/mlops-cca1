from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Student Performance Prediction API")

model = joblib.load("student_model.pkl")


class StudentData(BaseModel):
    Study_Hours: float
    Attendance: float
    Previous_Score: float


@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}


@app.post("/predict")
def predict(data: StudentData):
    input_data = [[
        data.Study_Hours,
        data.Attendance,
        data.Previous_Score
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }