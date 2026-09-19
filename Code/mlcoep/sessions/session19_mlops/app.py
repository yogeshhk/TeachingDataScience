# app.py     (run it with:  uvicorn app:app --port 8000)
# Step 2 of the "Deploy Your Own Model" section (LaTeX/ml_mlops_workflow.tex).
# Run train_model.py first: it creates mpg_model.joblib next to this file.
import os

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load(os.path.join(os.path.dirname(__file__), "mpg_model.joblib"))     # load once, at start-up
app = FastAPI()

class Car(BaseModel):
    engine_size: float
    horsepower: float
    weight: float

@app.post("/predict")
def predict(car: Car):
    x = [[car.engine_size, car.horsepower, car.weight]]
    return {"mpg_city": round(float(model.predict(x)[0]), 1)}
