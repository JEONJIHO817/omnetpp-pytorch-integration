# export_torchscript.py
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return torch.relu(self.fc(x))

# 모델 생성
model = TinyNet().eval()
example_input = torch.randn(1, 4)

# TorchScript로 변환 (jit.trace 방식)
traced = torch.jit.trace(model, example_input)

# TorchScript 모델 저장
traced.save("example.pt")

print("✅ TorchScript model saved as tinynet_traced.pt")
