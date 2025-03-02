아래는 요약과 함께 매우 상세하게 작성한 PyTorch 코드 예제입니다.  
이 코드는 직접 촬영한 이미지들을 클래스별 폴더 구조(ImageFolder 형식)로 구성한 데이터를 불러와, 데이터셋을 훈련/검증/테스트 세트로 나눈 후, timm 라이브러리의 MobileNetV4 모델을 사용하여 분류를 수행합니다.  
또한 tqdm 진행바, 에폭마다 손실 및 검증 정확도 그래프 기록, early stopping(최적 검증 성능 모델 저장 및 복원), 옵티마이저(Adam/AdamW) 선택, 가중치 감쇠 옵션(true/false) 선택, gradient clipping, 손실 튀는 경우 NaN 체크, 그리고 테스트 후 Accuracy, Precision, Recall, F1, elapsed time 등의 metric를 측정 및 출력하고 Confusion Matrix를 그리는 코드가 모두 포함되어 있습니다.

---

## 요약  
- **데이터셋 구성:** 직접 촬영한 이미지들을 ImageFolder 구조로 불러와 train/validation/test로 분할  
- **모델:** timm 라이브러리의 MobileNetV4 사용 (pretrained, 클래스 수에 맞게 출력층 수정)  
- **옵티마이저:** Adam과 AdamW 선택, 가중치 감쇠 옵션 제공  
- **학습:** tqdm 진행바, gradient clipping, 손실 튀는 경우 체크, early stopping 적용  
- **평가:** 테스트 시 Confusion Matrix 시각화 및 Accuracy, Precision, Recall, F1, elapsed time 측정

---

아래는 전체 코드입니다:

```python
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import timm  # timm 라이브러리를 통해 모델 생성
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1. 하이퍼파라미터 및 옵션 설정
# -----------------------------
data_dir = './data'  # 직접 촬영한 사진들을 클래스별로 하위 폴더에 넣은 경로 (예: data/class1, data/class2, ...)
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
patience = 5  # 검증 성능 개선이 없을 경우 기다릴 에폭 수
optimizer_choice = 'adam'  # 'adam' 또는 'adamw' 선택
use_weight_decay = True  # 가중치 감쇠 사용 여부 (True/False)
weight_decay_value = 1e-4 if use_weight_decay else 0.0  # 가중치 감쇠 값 (미사용 시 0)
max_grad_norm = 5.0  # gradient clipping 최대 norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

# -----------------------------
# 2. 데이터셋 및 데이터 로더 구성
# -----------------------------
# 학습 시 데이터 증강 적용, 검증/테스트 시에는 단순 전처리만 적용
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

# 직접 촬영한 이미지들을 폴더별로 분류해두었다고 가정하고 ImageFolder 사용
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# 데이터셋을 훈련, 검증, 테스트로 분할 (예: 70% 훈련, 15% 검증, 15% 테스트)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                        generator=torch.Generator().manual_seed(seed))

# 검증과 테스트 데이터에는 augmentation 없이 test_transform 적용
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -----------------------------
# 3. 모델 생성 (timm 이용, MobileNetV4)
# -----------------------------
num_classes = len(full_dataset.classes)
# pretrained 모델 불러오고, 마지막 FC 레이어를 num_classes에 맞게 설정
model = timm.create_model('mobilenetv4_100', pretrained=True, num_classes=num_classes)
model = model.to(device)

# -----------------------------
# 4. 손실 함수 및 옵티마이저 설정
# -----------------------------
criterion = nn.CrossEntropyLoss()

if optimizer_choice.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
elif optimizer_choice.lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
else:
    raise ValueError("optimizer_choice must be 'adam' or 'adamw'")

# -----------------------------
# 5. 학습, 검증, 테스트 함수 정의
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    
    # tqdm progress bar 적용
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Train Epoch {epoch}")
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 손실 값이 NaN이거나 너무 크면 업데이트 건너뛰기
        if torch.isnan(loss) or loss.item() > 1e3:
            print(f"Warning: Loss spike detected at batch {batch_idx} with loss {loss.item()}. Skipping update.")
            continue
        
        loss.backward()
        # Gradient clipping 적용
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss/(batch_idx+1))
    epoch_loss = running_loss / num_batches
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Validation")
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            # 예측 결과 계산
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            pbar.set_postfix(loss=running_loss/(batch_idx+1), accuracy=100.*correct/total)
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_preds)

# -----------------------------
# 6. 학습 루프 (Early Stopping 포함)
# -----------------------------
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
best_epoch = -1
early_stop_counter = 0
best_model_path = "best_model.pth"

for epoch in range(1, num_epochs+1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Early stopping 체크 (검증 손실 기준)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        early_stop_counter = 0
        # 검증 성능이 가장 좋을 때 모델 저장
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement for {early_stop_counter} epoch(s).")
    
    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

print(f"\nTraining finished. Best validation loss at epoch {best_epoch}.")

# -----------------------------
# 7. 저장된 모델 로드 및 테스트 수행 (평가 metric 측정)
# -----------------------------
model.load_state_dict(torch.load(best_model_path))

# 테스트 시작 시간 기록
start_time = time.time()
true_labels, pred_labels = test(model, test_loader, device)
elapsed_time = time.time() - start_time

# 평가 metric 계산 (멀티 클래스의 경우 macro average 사용)
test_accuracy  = accuracy_score(true_labels, pred_labels)
test_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
test_recall    = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
test_f1        = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

print("\n--- Test Metrics ---")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1 Score : {test_f1:.4f}")
print(f"Elapsed Time  : {elapsed_time:.2f} seconds")

# -----------------------------
# 8. Confusion Matrix 그리기
# -----------------------------
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
plt.title("Confusion Matrix on Test Data")
plt.show()

# -----------------------------
# 9. 학습 중 손실 및 정확도 그래프 그리기
# -----------------------------
epochs_range = range(1, len(train_losses)+1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 코드 주요 내용에 대한 추가 설명

1. **데이터셋 구성 및 전처리**  
   - `datasets.ImageFolder`를 통해 직접 촬영한 이미지를 불러오며, 데이터 증강은 학습용에 적용하고 검증/테스트용에는 단순 크기 조정 및 정규화만 수행합니다.
   - 전체 데이터셋을 70% 훈련, 15% 검증, 15% 테스트로 분할하여 사용합니다.

2. **모델 생성 및 옵티마이저**  
   - timm 라이브러리의 `mobilenetv4_100` 모델을 사용하며, pretrained 가중치를 불러와 최종 출력층을 데이터셋 클래스 수에 맞게 재구성합니다.
   - Adam과 AdamW 두 가지 옵티마이저 중 선택할 수 있으며, 가중치 감쇠 옵션을 통해 정규화를 적용할 수 있습니다.

3. **학습 루프 및 Early Stopping**  
   - 각 에폭마다 tqdm 진행바로 학습/검증 과정을 확인할 수 있으며, gradient clipping과 손실 NaN 체크를 통해 안정적인 학습을 도모합니다.
   - 검증 손실이 개선되지 않을 경우 early stopping을 적용하며, 최적 모델을 저장합니다.

4. **평가 및 성능 지표 측정**  
   - 테스트 단계에서 모델 예측 결과를 기반으로 Accuracy, Precision, Recall, F1 score를 계산하고, 테스트 소요 시간(elapsed time)을 측정합니다.
   - sklearn의 metric 함수들을 사용하며, 멀티 클래스 문제에서는 `average='macro'` 옵션을 적용합니다.
   - Confusion Matrix도 시각화하여 모델의 분류 성능을 직관적으로 확인할 수 있습니다.

5. **학습 및 검증 과정 시각화**  
   - 에폭 별 훈련 손실, 검증 손실 및 정확도 변화를 그래프로 출력하여 학습 과정을 모니터링합니다.

**참고 자료:**  
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)  
- [timm GitHub 저장소](https://github.com/rwightman/pytorch-image-models)  
- [tqdm GitHub](https://github.com/tqdm/tqdm)  

