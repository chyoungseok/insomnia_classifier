from my_dataset import gen_dataset
from my_model import SimpleSleepPPGModel_2

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 데이터셋 준비
print("Data loading ... ", end='')
full_dataset = gen_dataset(train_or_test='default')
print("done !")

# Train, Validation, Test Split
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # 나머지는 test에 할당
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 초기화
device = torch.device("cuda")
model = SimpleSleepPPGModel_2().to(device)

# 하이퍼파라미터 및 설정
num_epochs = 100
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Learning Rate Scheduler 설정 (Cosine Annealing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 학습 기록용 변수
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for signals, labels, masks in train_loader:  # mask 포함
        signals, labels, masks = signals.to(device), labels.to(device), masks.to(device)

        # Forward pass
        outputs = model(signals, masks)  # 모델에 mask 전달
        loss = criterion(outputs, labels)

        # Mask 적용: 패딩된 부분 무시
        # loss = (loss * masks).sum() / masks.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # Train Loss/Accuracy 계산
    train_loss = running_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation Loss/Accuracy 계산
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for signals, labels, masks in val_loader:  # mask 포함
            signals, labels, masks = signals.to(device), labels.to(device), masks.to(device)

            outputs = model(signals, masks)  # 모델에 mask 전달
            loss = criterion(outputs, labels)

            # Mask 적용: 패딩된 부분 무시
            # loss = (loss * masks).sum() / masks.sum()

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Learning Rate Scheduler 업데이트
    scheduler.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# 결과 시각화
epochs = range(1, num_epochs + 1)

# Loss 그래프
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.grid()

# Accuracy 그래프
plt.figure()
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.grid()

# Learning Rate 그래프
plt.figure()
plt.plot(epochs, learning_rates, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs. Epoch")
plt.legend()
plt.grid()

plt.show()
