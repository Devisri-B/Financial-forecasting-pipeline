from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib
from mangum import Mangum

app = FastAPI()

# Load Artifacts Globaly (Cold Start optimization)
ort_session = ort.InferenceSession("models/model.onnx")
scaler = joblib.load("models/scaler.pkl")
with open("models/calibration_score.txt", "r") as f:
    calibration_score = float(f.read())

class InputData(BaseModel):
    data: list # Expecting list of feature lists (seq_len x num_features)

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(payload: InputData):
    try:
        # Preprocessing
        input_arr = np.array(payload.data)
        
        # Reshape to (1, seq_len, features) if needed
        if len(input_arr.shape) == 2:
            input_arr = np.expand_dims(input_arr, axis=0)
            
        input_arr = input_arr.astype(np.float32)

        # Inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_arr}
        ort_outs = ort_session.run(None, ort_inputs)
        prediction = float(ort_outs[0][0][0])
        
        # Conformal Interval
        lower_bound = prediction - calibration_score
        upper_bound = prediction + calibration_score
        
        return {
            "prediction": prediction,
            "interval_95": [lower_bound, upper_bound],
            "confidence_gap": calibration_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Adapter for AWS Lambda
handler = Mangum(app)