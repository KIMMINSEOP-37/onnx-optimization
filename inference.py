import torch
import onnxruntime as ort
import numpy as np
import time
from train import CNN

# 테스트 입력
dummy_input = torch.randn(1, 1, 28, 28)

# ① PyTorch 추론 속도 측정
model = CNN()
model.load_state_dict(torch.load("models/cnn_mnist.pth"))
model.eval()

start = time.time()
for _ in range(1000):
    with torch.no_grad():
        _ = model(dummy_input)
pytorch_time = (time.time() - start) / 1000 * 1000  # ms

# ② ONNX Runtime 추론 속도 측정
sess = ort.InferenceSession("models/cnn_mnist.onnx")
input_np = dummy_input.numpy()

start = time.time()
for _ in range(1000):
    _ = sess.run(None, {"input": input_np})
onnx_time = (time.time() - start) / 1000 * 1000  # ms

# 결과 출력
print(f"PyTorch  추론 시간: {pytorch_time:.3f} ms")
print(f"ONNX     추론 시간: {onnx_time:.3f} ms")
print(f"속도 향상: {pytorch_time / onnx_time:.2f}x")