"""
AWS Lambda deployment handler for financial forecasting model.
Serverless inference endpoint for real-time predictions.
"""

import json
import torch
import numpy as np
import joblib
import base64
from io import BytesIO
import os

# Load model on cold start
MODEL_PATH = os.environ.get('MODEL_PATH', '/tmp/lstm_multi_ticker.pth')
SCALER_PATH = os.environ.get('SCALER_PATH', '/tmp/scaler_ensemble_multi.pkl')

# Global model and scaler (cached in Lambda memory)
model = None
scaler = None

def load_model():
    """Load model and scaler from S3/local storage."""
    global model, scaler
    
    if model is None:
        try:
            # For production: Load from S3
            # import boto3
            # s3 = boto3.client('s3')
            # s3.download_file('my-bucket', 'models/lstm_multi_ticker.pth', MODEL_PATH)
            
            model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    return model, scaler

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def handler(event, context):
    """
    AWS Lambda handler for stock price predictions.
    
    Expected input:
    {
        "ticker": "SPY",
        "features": [array of 10 features],
        "action": "predict"
    }
    
    Returns:
    {
        "prediction": float,
        "confidence": float,
        "timestamp": str
    }
    """
    
    try:
        # Parse input
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event
        
        ticker = body.get('ticker', 'UNKNOWN')
        features = body.get('features', [])
        
        if not features or len(features) != 60*10:  # 60 timesteps x 10 features
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid input. Expected 60x10 features (600 values total)'
                })
            }
        
        # Load model
        model, scaler = load_model()
        
        # Reshape features to (1, 60, 10)
        features_array = np.array(features).reshape(1, 60, 10)
        
        # Normalize
        features_scaled = scaler.transform(
            features_array.reshape(-1, 10)
        ).reshape(1, 60, 10)
        
        # Predict
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            prediction = model(features_tensor).numpy()[0, 0]
        
        # Inverse transform to original scale
        pred_full = np.zeros((1, 10))
        pred_full[0, 0] = prediction
        pred_original = scaler.inverse_transform(pred_full)[0, 0]
        
        # Calculate confidence (mock - in production would use ensemble uncertainty)
        confidence = 0.9826  # Placeholder: Use actual model R²
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'ticker': ticker,
                'prediction': float(pred_original),
                'confidence': confidence,
                'model': 'LSTM Multi-Ticker Ensemble',
                'timestamp': context.aws_request_id if context else 'unknown'
            })
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# For local testing
if __name__ == "__main__":
    class MockContext:
        aws_request_id = "test-request-id"
    
    test_event = {
        'ticker': 'SPY',
        'features': np.random.randn(600).tolist()  # 60x10
    }
    
    response = handler(test_event, MockContext())
    print(json.dumps(json.loads(response['body']), indent=2))
