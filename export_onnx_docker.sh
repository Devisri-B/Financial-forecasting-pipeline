#!/bin/bash
# Quick ONNX export using Docker (fixes Mac→Lambda compatibility)
# Use this if you already have lstm_multi_ticker.pth trained

set -e

echo "🔧 Exporting Multi-Ticker LSTM to ONNX (Linux x86_64)"
echo "======================================================"
echo ""

# Check if model exists
if [ ! -f "models/lstm_multi_ticker.pth" ]; then
    echo "❌ Error: models/lstm_multi_ticker.pth not found"
    echo ""
    echo "Please train the model first:"
    echo "  env/bin/python src/train_ensemble_multi.py"
    echo ""
    echo "Or use full deployment script:"
    echo "  ./deploy_to_aws.sh"
    exit 1
fi

echo "✅ Found existing model: models/lstm_multi_ticker.pth"
echo ""

# Build training Docker image (has PyTorch + ONNX)
echo "📦 Building Docker image..."
docker build --platform linux/amd64 -f Dockerfile.train -t financial-trainer:latest . > /dev/null 2>&1

echo "✅ Docker image built"
echo ""

# Export ONNX inside Docker
echo "🔄 Exporting to ONNX (Linux x86_64)..."
docker run --platform linux/amd64 \
  --rm \
  -v "$(pwd)/models:/var/task/models" \
  -v "$(pwd)/src:/var/task/src" \
  financial-trainer:latest \
  python -c "
import torch
import sys
sys.path.insert(0, '/var/task')

from src.model import StockPredictor

# Load model
model = StockPredictor(input_dim=12, hidden_dim=128, num_layers=3, dropout=0.05)
model.load_state_dict(torch.load('/var/task/models/lstm_multi_ticker.pth', map_location='cpu'))
model.eval()

# Export
dummy_input = torch.randn(1, 60, 12)
torch.onnx.export(
    model,
    dummy_input,
    '/var/task/models/model_fixed.onnx',
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print('✅ ONNX exported: /var/task/models/model_fixed.onnx')
"

echo ""
echo "🔍 Verifying ONNX model..."
docker run --platform linux/amd64 \
  --rm \
  -v "$(pwd)/models:/var/task/models" \
  financial-trainer:latest \
  python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('/var/task/models/model_fixed.onnx')
print(f'  Input shape: {sess.get_inputs()[0].shape}')
print(f'  Output shape: {sess.get_outputs()[0].shape}')

# Test
test_input = np.random.randn(1, 60, 12).astype(np.float32)
ort_inputs = {sess.get_inputs()[0].name: test_input}
ort_outputs = sess.run(None, ort_inputs)
print(f'  Test prediction: {ort_outputs[0][0][0]:.4f}')
"

echo ""
echo "✅ ONNX export complete and verified!"
echo ""
echo "📝 Next steps:"
echo "1. Test locally: python test_lambda_local.py"
echo "2. Build Lambda image: docker build --platform linux/amd64 -t financial-forecaster ."
echo "3. Deploy to AWS via GitHub Actions or manual push"
echo ""
