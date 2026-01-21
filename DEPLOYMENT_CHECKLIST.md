# Complete Deployment Checklist

## ✅ What You Already Have

- [x] GitHub Actions workflow (`.github/workflows/deploy.yml`)
- [x] Dockerfile for Lambda (`Dockerfile`)
- [x] Dockerfile for Training (`Dockerfile.train`)
- [x] AWS Lambda function created
- [x] AWS ECR repository created
- [x] Function URL configured
- [x] IAM user with ECR + Lambda permissions
- [x] GitHub secrets configured (AWS credentials)
- [x] Multi-ticker data downloaded (7,666 samples)

## 🚨 Critical Issue Fixed

**Problem**: Mac (especially M1/M2/M3) exports ONNX models with architecture-specific optimizations that fail on AWS Lambda (Linux x86_64).

**Symptoms**:
- `ImportError`: NumPy version mismatch
- `ONNXRuntimeError`: Model version mismatch
- Model works locally but fails on Lambda

**Solution**: Use Docker with `--platform linux/amd64` to ensure compatibility.

---

## 🚀 Deployment Steps

### Option A: Quick Deployment (Use Existing Model)

If you already have `models/lstm_multi_ticker.pth` trained:

```bash
# 1. Export ONNX in Docker (ensures Linux x86_64 compatibility)
./export_onnx_docker.sh

# 2. Test locally
python test_lambda_local.py

# 3. Push to GitHub (triggers GitHub Actions)
git add .
git commit -m "Deploy multi-ticker model to Lambda"
git push origin main

# GitHub Actions will automatically:
#   - Build Docker image
#   - Push to ECR
#   - Update Lambda function
```

---

### Option B: Full Pipeline (Train + Deploy)

Train multi-ticker model from scratch in Docker:

```bash
# Complete training + deployment in Docker
./deploy_to_aws.sh

# This will:
# 1. Train multi-ticker ensemble (7,666 samples, R²=0.9986)
# 2. Export ONNX (Linux x86_64)
# 3. Verify ONNX works
# 4. Test Lambda handler locally
# 5. Build Lambda Docker image
# 6. Show manual AWS deployment commands
```

---

### Option C: Manual Step-by-Step

#### Step 1: Train Multi-Ticker Model

```bash
# Train in Docker to ensure compatibility
docker build --platform linux/amd64 -f Dockerfile.train -t financial-trainer .

docker run --platform linux/amd64 \
  -v "$(pwd)/models:/var/task/models" \
  -v "$(pwd)/data:/var/task/data" \
  financial-trainer \
  python src/train_ensemble_multi.py
```

**Expected output**:
- `models/lstm_multi_ticker.pth` (PyTorch weights)
- `models/scaler_ensemble_multi.pkl` (StandardScaler)
- R² ≈ 0.9826 for LSTM
- R² ≈ 0.9986 for Ensemble

#### Step 2: Export to ONNX (Linux x86_64)

```bash
./export_onnx_docker.sh
```

**Verifies**:
- Input shape: `['batch_size', 60, 12]` ✅ (12 features for multi-ticker)
- Output shape: `['batch_size', 1]`
- Test prediction works

#### Step 3: Test Lambda Locally

```bash
python test_lambda_local.py
```

**Expected**:
```json
{
  "prediction": 152.45,
  "interval_95": [150.95, 153.95],
  "confidence_gap": 1.5507
}
```

#### Step 4: Build Lambda Docker Image

```bash
docker build --platform linux/amd64 -t financial-forecaster:latest .
```

**This includes**:
- FastAPI app (`src/app.py`)
- ONNX model (`models/model_fixed.onnx` - 12 features)
- Scaler (`models/scaler_ensemble_multi.pkl`)
- Calibration score (`models/calibration_score.txt`)

#### Step 5: Push to AWS ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag financial-forecaster:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/financial-forecaster:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/financial-forecaster:latest
```

#### Step 6: Update Lambda Function

```bash
aws lambda update-function-code \
  --function-name financial-forecaster \
  --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/financial-forecaster:latest
```

**Wait ~30 seconds** for Lambda to pull and deploy new image.

#### Step 7: Test Function URL

```bash
curl -X POST https://<your-function-url>.lambda-url.us-east-1.on.aws/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [150.2, 151.3, 149.8, 150.9, 1000000, 0.5, 0.3, -0.2, 150.5, 151.0, 0.6, 0.4],
      ... (60 timesteps total)
    ]
  }'
```

**Expected response**:
```json
{
  "prediction": 152.45,
  "interval_95": [150.95, 153.95],
  "confidence_gap": 1.5507
}
```

---

## 🔍 Troubleshooting

### Issue: ONNX Shape Mismatch

**Error**: `Expected: 10, Got: 12`

**Cause**: Using old single-ticker ONNX model (10 features) instead of multi-ticker (12 features)

**Fix**:
```bash
./export_onnx_docker.sh  # Re-export in Docker
```

### Issue: NumPy Version Mismatch

**Error**: `ImportError: NumPy version mismatch`

**Cause**: Mac NumPy (arm64) != Lambda NumPy (x86_64)

**Fix**: Always use Docker with `--platform linux/amd64`

### Issue: Lambda Timeout

**Error**: Function times out during cold start

**Fix**: Increase Lambda timeout to 30 seconds (default is 3s)
```bash
aws lambda update-function-configuration \
  --function-name financial-forecaster \
  --timeout 30
```

### Issue: ONNX Model Too Large

**Error**: Image size > 10GB

**Fix**: Already optimized:
- Using CPU-only PyTorch
- ONNX model is ~1.3MB (small!)
- No GPU dependencies

---

## 📊 Validation Checklist

After deployment, verify:

- [ ] Health check works: `curl https://<url>/`
- [ ] Prediction endpoint works: `curl -X POST https://<url>/predict`
- [ ] Response has correct structure (`prediction`, `interval_95`, `confidence_gap`)
- [ ] Prediction value is reasonable (stock price range)
- [ ] Latency < 200ms (cold start < 3s)
- [ ] CloudWatch logs show no errors

---

## 🎯 What This Achieves

**For Recruiters**:
- ✅ Production deployment on AWS Lambda
- ✅ CI/CD pipeline via GitHub Actions
- ✅ Docker expertise (multi-stage, platform-specific builds)
- ✅ ONNX model optimization for serverless
- ✅ Cost optimization ($0.35/month for 100 daily predictions)

**Technical Stats**:
- **Model**: Multi-ticker LSTM ensemble (R²=0.9986)
- **Input**: 60 timesteps × 12 features
- **Output**: Price prediction + 95% confidence interval
- **Latency**: ~50-100ms (warm) / ~2-3s (cold start)
- **Cost**: ~$0.000017 per prediction

---

## 📝 Next Steps After Deployment

1. **Monitoring**: Set up CloudWatch alarms for errors/latency
2. **Load testing**: Use Apache Bench or Locust to test concurrency
3. **Versioning**: Tag releases (v1.0, v1.1, etc.) in ECR
4. **Model updates**: Retrain monthly with new market data
5. **Dashboard**: Build Streamlit/Gradio frontend
6. **API docs**: Add OpenAPI/Swagger documentation

---

## 💡 Key Learnings

1. **Platform matters**: Mac ONNX != Linux ONNX (always use Docker)
2. **NumPy is tricky**: Version must match exactly between train and inference
3. **GitHub Actions rocks**: Push code → auto-deploy (no manual steps)
4. **Lambda is cheap**: $0.35/month beats $100/month EC2 server
5. **ONNX is fast**: ~50ms inference (vs ~200ms for PyTorch)

Always check the correctness of AI-generated responses.
