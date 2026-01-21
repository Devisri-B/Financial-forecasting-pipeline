# AWS Lambda Deployment Guide

## Overview
Production-ready serverless deployment of financial forecasting model on AWS Lambda.

## Architecture
```
User API Request
    ↓
API Gateway (HTTPS endpoint)
    ↓
Lambda Function (lambda_handler.py)
    ↓
Model Inference (CPU, <100ms)
    ↓
JSON Response with Prediction
```

## Step 1: Prepare Model Package

### Create deployment package:
```bash
# Create deployment directory
mkdir lambda_deployment
cd lambda_deployment

# Copy model files
cp ../models/lstm_multi_ticker.pth .
cp ../models/scaler_ensemble_multi.pkl .
cp ../src/lambda_handler.py .

# Install dependencies
pip install -r requirements.txt -t .

# Create ZIP for upload
zip -r lambda_function.zip .
```

### Required dependencies (requirements.txt):
```
torch==2.0.0
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
joblib==1.3.2
```

## Step 2: Create AWS Lambda Function

### Via AWS Console:
1. **Services** → **Lambda** → **Create Function**
2. **Function name**: `financial-forecasting-lstm`
3. **Runtime**: Python 3.11
4. **Architecture**: x86_64
5. **Create function**

### Upload code:
1. **Code source** → **Upload from** → **.zip file**
2. Select `lambda_function.zip`
3. **Deploy**

### Configure settings:
- **Memory**: 512 MB (sufficient for inference)
- **Timeout**: 30 seconds
- **Ephemeral storage**: 512 MB

## Step 3: Create API Gateway

### Create REST API:
1. **Services** → **API Gateway** → **Create API**
2. **REST API** → **Build**
3. **API name**: `financial-forecasting-api`

### Create resource and method:
1. **Resources** → **Create Resource**
   - **Resource name**: `predict`
   - **Create resource**

2. **POST method** → **Lambda Function**
   - **Lambda Function**: `financial-forecasting-lstm`
   - **Save**

3. **CORS** → **Enable CORS**
   - **Save**

### Deploy API:
1. **Deploy API**
   - **Deployment stage**: `prod`
   - **Deploy**

2. Copy **Invoke URL**: `https://xxxxx.execute-api.region.amazonaws.com/prod`

## Step 4: Test API

### Using curl:
```bash
curl -X POST \
  https://xxxxx.execute-api.region.amazonaws.com/prod/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "SPY",
    "features": [array of 600 normalized values]
  }'
```

### Response:
```json
{
  "ticker": "SPY",
  "prediction": 450.25,
  "confidence": 0.9826,
  "model": "LSTM Multi-Ticker Ensemble",
  "timestamp": "12345-request-id"
}
```

## Step 5: Cost Management

### Lambda pricing (US East - N. Virginia):
- **Compute**: $0.0000002 per 100ms
- **Requests**: $0.20 per million requests
- **Free tier**: 1 million requests/month + 400,000 GB-seconds

### Example costs:
- 100 daily predictions (100ms each):
  - $0.20/month compute
  - $0.06/month requests
  - **Total: ~$0.30/month**

### API Gateway pricing:
- **Per request**: $3.50 per million requests
- 100 daily requests = ~$0.01/month

### Total monthly cost (100 daily predictions):
**~$0.35/month** (within free tier!)

## Cost Optimization Tips

1. **Keep Lambda memory at 512MB** - Too low causes timeouts, too high wastes money
2. **Monitor via CloudWatch** - Set alarms for unexpected costs
3. **Disable API if not used** - Delete API Gateway to avoid unexpected charges
4. **Use S3 for model storage** - S3 is cheaper than Lambda ephemeral storage

## Production Considerations

### Security:
```python
# Add API key authentication
import json

def require_api_key(event):
    api_key = event.get('headers', {}).get('X-API-Key')
    if api_key != os.environ['API_KEY']:
        return {
            'statusCode': 401,
            'body': json.dumps({'error': 'Unauthorized'})
        }
    return None
```

### Monitoring:
- **CloudWatch Logs**: Monitor errors and performance
- **X-Ray**: Trace requests for debugging
- **Alarms**: Alert on high error rates

### Deployment Best Practices:
1. Use **SAM (Serverless Application Model)** for infrastructure as code
2. Keep model in **S3** for easier updates
3. Use **Lambda layers** for shared dependencies
4. Implement **versioning** for model updates

## Stopping the API (Cost Control)

### To save costs when not using:
```bash
# Delete API Gateway
aws apigateway delete-rest-api --rest-api-id xxxxx

# Disable Lambda (or delete)
aws lambda delete-function --function-name financial-forecasting-lstm
```

### Restore:
Keep CloudFormation stack → Export → Re-import to restore quickly

## Code Examples

### Python client:
```python
import requests
import json

API_URL = "https://xxxxx.execute-api.region.amazonaws.com/prod/predict"

def predict(ticker, features):
    response = requests.post(
        API_URL,
        json={"ticker": ticker, "features": features},
        headers={"X-API-Key": "your-api-key"}
    )
    return response.json()

# Usage
result = predict("SPY", my_features)
print(f"Predicted price: ${result['prediction']:.2f}")
```

### Using AWS SDK:
```bash
aws lambda invoke \
  --function-name financial-forecasting-lstm \
  --payload '{"ticker":"SPY","features":[...]}' \
  response.json

cat response.json
```

---

## Screenshots & Documentation

### MLflow Experiment Tracking
See `MLFLOW_SHOWCASE.md` for detailed screenshots showing:
- Experiment runs (A, B, C)
- Metrics comparison
- Parameter configurations
- Historical run tracking

### Next Steps
1. Deploy Lambda function
2. Test API endpoints
3. Monitor CloudWatch logs
4. Scale based on demand

---

**Status**: ✅ Production-ready  
**Cost**: ~$0.35/month for 100 daily predictions  
**Model R²**: 0.9826 (LSTM), 0.9986 (Ensemble)
