from my_dataset import gen_dataset
from my_model import SimpleSleepPPGModel

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 데이터셋 준비
train_dataset = gen_dataset(train_or_test='train')
test_dataset = gen_dataset(train_or_test='test')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

my_model = SimpleSleepPPGModel()

# 하이퍼파라미터
num_epochs = 100
learning_rate = 0.001

# 모델, 손실 함수, 옵티마이저 초기화
device = torch.device("cuda")
model = SimpleSleepPPGModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for signals, labels in train_loader:
        signals, labels = signals.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
