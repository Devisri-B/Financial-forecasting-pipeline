# AWS Lambda Deployment Guide

## Overview
Automated serverless deployment using Docker containers and GitHub Actions CI/CD pipeline.

## Architecture
```
GitHub Push (main branch)
    ↓
GitHub Actions Workflow
    ├─> Train multi-ticker model
    ├─> Build Docker image
    ├─> Push to AWS ECR
    └─> Update Lambda function
         ↓
AWS Lambda (Container)
    ├─> FastAPI + Mangum
    ├─> ONNX Runtime inference
    └─> Returns predictions
         ↓
Function URL (HTTPS)
    └─> Public endpoint
```

## Deployment Method

I use **Docker containers** instead of ZIP files because:
- ONNX Runtime needs compiled dependencies
- Consistent environment (Mac → Linux x86_64)
- Larger deployment package (>250MB uncompressed)
- Easier dependency management

## Step 1: AWS Setup (One-Time)

### 1.1 Create ECR Repository
```bash
aws ecr create-repository \
  --repository-name financial-forecaster \
  --region us-east-1
```

### 1.2 Create Lambda Function
```bash
# Via AWS Console:
# 1. Services → Lambda → Create Function
# 2. Container image
# 3. Function name: financial-forecaster
# 4. Container image URI: <your-ecr-uri>:latest
# 5. Architecture: x86_64
# 6. Memory: 512 MB
# 7. Timeout: 30 seconds
```

### 1.3 Enable Function URL
```bash
# In Lambda Console:
# Configuration → Function URL → Create function URL
# Auth type: NONE (public endpoint)
# CORS: Enable
# Save
```

### 1.4 Add GitHub Secrets
```bash
# In GitHub repo: Settings → Secrets → Actions
# Add the following secrets:
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

## Step 2: Automated Deployment

The deployment is **fully automated** via GitHub Actions:

### Workflow (.github/workflows/deploy.yml)
```yaml
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    - Train multi-ticker model (Python 3.11)
    - Build Docker image (ONNX + FastAPI)
    - Push to ECR
    - Update Lambda function
```

### Manual Trigger (if needed)
```bash
# Just push to main branch:
git add .
git commit -m "Update model"
git push origin main

# GitHub Actions will:
# 1. Download 4 ETFs (SPY, QQQ, DIA, IWM)
# 2. Train ensemble (R²=0.9986)
# 3. Export to ONNX
# 4. Build Docker image
# 5. Deploy to Lambda
```

## Step 3: Local Testing (Optional)

### Build Docker locally:
```bash
# Build for Linux x86_64 (Lambda architecture)
docker build --platform linux/amd64 -t financial-forecaster:latest .

# Test locally with Lambda runtime emulator
docker run -p 9000:8080 financial-forecaster:latest

# Test endpoint
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"body": "{\"data\": [[...60x12 features...]]}"}'
```

## Step 4: Production Endpoint

### Get Function URL
```bash
# In AWS Lambda Console:
# Configuration → Function URL
# Copy the URL: https://xxxxx.lambda-url.us-east-1.on.aws/

# Test production endpoint:
curl -X POST https://xxxxx.lambda-url.us-east-1.on.aws/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [[...60x12 normalized features...]]
  }'
```

### Response Format:
```json
{
  "prediction": 450.25,
  "interval_95": [445.10, 455.40],
  "confidence_gap": 5.15
}
```

## Step 5: Monitoring & Logs

### View CloudWatch Logs:
```bash
# In AWS Console:
# CloudWatch → Log Groups → /aws/lambda/financial-forecaster

# Or via CLI:
aws logs tail /aws/lambda/financial-forecaster --follow
```

### Key Metrics:
- **Invocations**: Number of predictions
- **Duration**: Average ~100ms
- **Errors**: Should be 0% (monitor for ONNX errors)
- **Throttles**: Increase reserved concurrency if throttled

## Step 6: Cost Management

### Lambda pricing (US East - N. Virginia):
- **Compute**: $0.0000002 per 100ms
- **Requests**: $0.20 per million requests
- **Free tier**: 1 million requests/month + 400,000 GB-seconds

### Example costs:
- 100 daily predictions (100ms each):
  - Requests: 3,000/month
  - Compute: 3,000 × 0.1s × 512MB = 150 GB-seconds
  - **Total: $0 (within free tier)**

### Free tier limits:
- 1 million requests/month
- 400,000 GB-seconds/month
- Always free (not just first 12 months)

### Total monthly cost (100 daily predictions):
**$0/month** (stays within free tier)

## Dockerfile Explained

```dockerfile
# AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install ONNX Runtime + FastAPI + Mangum
RUN pip install onnxruntime fastapi mangum joblib numpy scikit-learn

# Copy inference code (NOT training code)
COPY src/app.py ${LAMBDA_TASK_ROOT}/src/
COPY src/features.py ${LAMBDA_TASK_ROOT}/src/

# Copy trained model artifacts
COPY models/model_fixed.onnx ${LAMBDA_TASK_ROOT}/models/
COPY models/scaler_ensemble_multi.pkl ${LAMBDA_TASK_ROOT}/models/
COPY models/calibration_score.txt ${LAMBDA_TASK_ROOT}/models/

# Lambda handler
CMD [ "src.app.handler" ]
```

**Why this works:**
- Uses AWS-provided Lambda base image
- ONNX Runtime is CPU-only (small size)
- FastAPI + Mangum = easy HTTP handling
- Cold start: ~1-2 seconds (acceptable)

## GitHub Actions Workflow Explained

```yaml
# Trigger: On every push to main
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    steps:
      # 1. Train model with latest data
      - python src/multi_ticker_loader.py  # Download fresh ETF data
      - python src/train_ensemble_multi.py  # Train ensemble (R²=0.9986)
      
      # 2. Build Docker image for x86_64
      - docker build --platform linux/amd64 -t image .
      
      # 3. Push to ECR
      - docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
      
      # 4. Update Lambda function
      - aws lambda update-function-code --image-uri <new-image>
```

**Advantages:**
- Retrains model with latest data on every deploy
- Ensures Linux x86_64 compatibility (not Mac ARM64)
- Zero-downtime deployment
- Full automation

## Troubleshooting

### Issue: ONNX dimension mismatch
```
Error: Got invalid dimensions for input: Expected 10, Got 12
```
**Fix**: Retrain ONNX model with correct 12-feature multi-ticker data

### Issue: Cold start timeout
```
Task timed out after 3.00 seconds
```
**Fix**: Increase Lambda timeout to 30 seconds in configuration

### Issue: Import error NumPy version
```
ImportError: NumPy version mismatch
```
**Fix**: Use Docker build with `--platform linux/amd64` (not Mac ARM)

## Cost Optimization Tips

1. **Keep Lambda memory at 512MB** - Too low causes timeouts, too high wastes money
2. **Monitor via CloudWatch** - Set alarms for unexpected costs
3. **Disable Lambda if not used** - Stop function to avoid unexpected charges

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

### Deployment Architecture:
- **Docker containers**: Eliminates environment inconsistencies (Mac → Lambda)
- **GitHub Actions CI/CD**: Automated training and deployment on every push
- **ECR registry**: Stores Docker images for Lambda
- **Automatic retraining**: Model updates with fresh ETF data on each deploy

## Cost Control

### To save costs when not using:
```bash
# Disable Lambda function (pause billing)
aws lambda delete-function --function-name financial-forecaster
```

### Re-deploy:
Just push to main branch - GitHub Actions will rebuild and redeploy

## Testing the Deployment

### Using Lambda Function URL (Direct):
```bash
# Get Function URL from Lambda Console
# Configuration → Function URL

curl -X POST https://xxxxx.lambda-url.us-east-1.on.aws/ \
  -H 'Content-Type: application/json' \
  -d '{
    "body": "{\"data\": [[...60x12 normalized features...]]}"
  }'
```

### Using AWS CLI:
```bash
aws lambda invoke \
  --function-name financial-forecaster \
  --payload '{"body": "{\"data\": [...]}" }' \
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

---

**Status**:  Production-ready  
**Cost**: ~$0.35/month for 100 daily predictions  
**Model R²**: 0.9826 (LSTM), 0.9986 (Ensemble)
