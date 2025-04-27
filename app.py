import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input data
class HeartDiseaseData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load model when app starts
try:
    model = joblib.load('heart_disease_model.pkl')
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a model.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Heart Disease Detection API!"}

# Prediction endpoint
@app.post('/predict')
async def predict(data: HeartDiseaseData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or invalid model file.")
    
    try:
        # Prepare input data
        input_array = np.array([[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]])

        # Predict
        prediction = model.predict(input_array)
        result = int(prediction[0])  # Convert numpy int to normal int for JSON serialization

        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")




