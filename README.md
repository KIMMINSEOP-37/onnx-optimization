# ONNX Optimization: PyTorch vs ONNX Runtime

Comparing inference speed between PyTorch and ONNX Runtime on a CNN model trained with MNIST dataset.

## Results

| Model | Inference Time | Speedup |
|-------|---------------|---------|
| PyTorch | 0.336 ms | 1.00x |
| ONNX Runtime | 0.097 ms | 3.47x |

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