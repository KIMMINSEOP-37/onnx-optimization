import torch
import torch.onnx
from train import CNN

# 모델 불러오기
model = CNN()
model.load_state_dict(torch.load("models/cnn_mnist.pth"))
model.eval()

# ONNX로 변환
dummy_input = torch.randn(1, 1, 28, 28)  # MNIST 이미지 크기
torch.onnx.export(
    model,
    dummy_input,
    "models/cnn_mnist.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX 변환 완료 → models/cnn_mnist.onnx")