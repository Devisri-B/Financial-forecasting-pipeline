"""
Export multi-ticker LSTM model to ONNX format for AWS Lambda deployment.
This fixes the feature dimension mismatch (10 vs 12 features).
"""

import torch
import os

def export_multi_ticker_to_onnx():
    """Export the trained multi-ticker LSTM model to ONNX format"""
    
    # Load the trained multi-ticker model
    model_path = "models/lstm_multi_ticker.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Run this first: python src/train_ensemble_multi.py")
        return
    
    # Import model class
    from src.model import StockPredictor
    
    # Model configuration (from multi-ticker training)
    input_dim = 12  # Multi-ticker has 12 features
    hidden_dim = 128
    num_layers = 3
    dropout = 0.05
    
    # Initialize model
    model = StockPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    print(f"   Input dim: {input_dim} features")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Layers: {num_layers}")
    
    # Create dummy input (batch_size=1, seq_len=60, features=12)
    dummy_input = torch.randn(1, 60, input_dim)
    
    # Export to ONNX
    output_path = "models/model_fixed.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\n✅ Exported to {output_path}")
    
    # Verify the export
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    input_shape = sess.get_inputs()[0].shape
    output_shape = sess.get_outputs()[0].shape
    
    print(f"   Input shape: {input_shape}")
    print(f"   Output shape: {output_shape}")
    
    # Test with dummy data
    test_input = dummy_input.numpy()
    ort_inputs = {sess.get_inputs()[0].name: test_input}
    ort_outputs = sess.run(None, ort_inputs)
    
    print(f"\n✅ ONNX model verified!")
    print(f"   Test prediction: {ort_outputs[0][0][0]:.4f}")
    
    return output_path

if __name__ == "__main__":
    export_multi_ticker_to_onnx()
