import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. CSV 로드
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1).values.astype(np.float32)
y = df["Outcome"].values.astype(np.float32)

# 2. Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("mean =", scaler.mean_.tolist())
print("scale =", scaler.scale_.tolist())


# numpy → torch 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader 구성
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 4. MLP 모델 정의
class DiabetesClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = DiabetesClassifier()

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 학습 루프
epochs = 200
for epoch in range(epochs):
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 6. 테스트 정확도
model.eval()
with torch.no_grad():
    pred = model(X_test)
    pred_labels = (pred > 0.5).float()
    accuracy = (pred_labels == y_test).float().mean().item()

print("\nTest Accuracy:", accuracy)

# 7. TorchScript 변환
example_input = torch.randn(1, 8)
scripted = torch.jit.trace(model, example_input)
scripted.save("diabetes_clf.pt")

print("\nSaved TorchScript model as diabetes_clf.pt")
