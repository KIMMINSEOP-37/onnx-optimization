# ONNX Optimization: PyTorch vs ONNX Runtime

## Background
PyTorch는 학습에 최적화된 프레임워크로, 추론 시 불필요한 연산 오버헤드가 발생합니다. 반면 ONNX Runtime은 추론만을 위해 설계된 경량 실행 엔진으로, 동일한 모델을 더 빠르게 실행할 수 있습니다. 이 프로젝트는 PyTorch 모델을 ONNX 형식으로 변환하고, 두 실행 환경 간의 추론 속도 차이를 실제로 측정합니다.





## Results
| Model         | Inference Time | Speedup |
|---------------|----------------|---------|
| PyTorch       | 0.336 ms       | 1.00x   |
| ONNX Runtime  | 0.097 ms       | 3.47x   |


## Project Structure

```
onnx-optimization/
├── train.py        # CNN model training on MNIST
├── export.py       # Export PyTorch model to ONNX format
├── inference.py    # Inference speed comparison
└── models/
    ├── cnn_mnist.pth    # PyTorch model
    └── cnn_mnist.onnx   # ONNX model
```
## Pipeline

```mermaid
flowchart LR
    A[MNIST Dataset] --> B[CNN Training]
    B --> C[cnn_mnist.pth]
    C --> D[ONNX Export]
    D --> E[cnn_mnist.onnx]
    E --> F[Inference Comparison]
    F --> G[PyTorch: 0.336ms]
    F --> H[ONNX Runtime: 0.097ms]
    G --> I[3.47x Speedup ⚡]
    H --> I
```

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

