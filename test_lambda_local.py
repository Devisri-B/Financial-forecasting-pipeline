"""
Test Lambda function locally before deploying to AWS.
Ensures model loads correctly and predictions work.
"""

import json
import numpy as np

def test_fastapi_directly():
    """Test FastAPI app directly (before Mangum wrapping)"""
    from src.app import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Create sample input (60 timesteps x 12 features)
    sample_data = np.random.randn(60, 12).tolist()
    
    # Test health check
    health_response = client.get("/")
    print("✅ Health check:", health_response.json())
    
    # Test prediction
    prediction_response = client.post(
        "/predict",
        json={"data": sample_data}
    )
    
    print("\n✅ Prediction Response:")
    print(json.dumps(prediction_response.json(), indent=2))
    
    # Validate response
    assert prediction_response.status_code == 200, f"Got status {prediction_response.status_code}"
    body = prediction_response.json()
    assert 'prediction' in body, "Missing 'prediction' key"
    assert 'interval_95' in body, "Missing 'interval_95' key"
    assert 'confidence_gap' in body, "Missing 'confidence_gap' key"
    
    print("\n✅ All tests passed! Lambda is ready for deployment.")
    return prediction_response.json()

if __name__ == "__main__":
    test_fastapi_directly()
