# ONNX Optimization: PyTorch vs ONNX Runtime

Comparing inference speed between PyTorch and ONNX Runtime on a CNN model trained with MNIST dataset.

## Motivation
On-device AI requires fast inference on resource-constrained hardware.
This project explores how ONNX Runtime can accelerate inference 
compared to standard PyTorch.


## Project Structure
onnx-optimization/
├── train.py        # CNN model training on MNIST
├── export.py       # Export PyTorch model to ONNX format
├── inference.py    # Inference speed comparison
└── models/
├── cnn_mnist.pth   # PyTorch model
└── cnn_mnist.onnx  # ONNX model

## Environment

- Python 3.x
- PyTorch
- ONNX Runtime

## How to Run

**1. Train the model**
```bash
python train.py
```

**2. Export to ONNX**
```bash
python export.py
```

**3. Compare inference speed**
```bash
python inference.py
```

## Key Findings
ONNX Runtime achieved **3.47x faster inference** than PyTorch.
This demonstrates the potential of model format conversion 
for edge deployment scenarios.

## Results

| Model | Inference Time | Speedup |
|-------|---------------|---------|
| PyTorch | 0.336 ms | 1.00x |
| ONNX Runtime | 0.097 ms | 3.47x |
