아래는 요약과 함께 매우 상세하게 작성한 PyTorch 코드 예제입니다.  
이 코드는 직접 촬영한 이미지들을 폴더 구조(ImageFolder 형식)로 구성한 데이터를 불러와, 데이터셋을 훈련/검증/테스트 세트로 나눈 후, timm 라이브러리의 MobileNetV4 모델을 사용하여 분류를 수행합니다.  
또한 tqdm을 활용한 진행바, 에폭마다 훈련/검증 손실 그래프 기록, 테스트 결과로 confusion matrix 그리기, Adam/AdamW 옵티마이저 선택, 가중치 감쇠 옵션(true/false) 선택, 그리고 손실 값이 갑자기 튀는 경우를 대비한 gradient clipping 및 NaN 체크 기능, early stopping 기능(검증 성능 개선 없을 시 중단 후 최고 성능 모델 저장 및 복원)까지 모두 포함되어 있습니다.

---

## 상세 코드 설명

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import timm  # timm 라이브러리를 통해 모델 생성
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1. 하이퍼파라미터 및 옵션 설정
# -----------------------------
data_dir = './data'  # 직접 촬영한 사진들을 클래스별로 하위 폴더에 넣은 경로 (예: data/class1, data/class2, ...)
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
patience = 5  # 검증 성능 개선이 없을 경우 몇 에폭 기다릴지
optimizer_choice = 'adam'  # 'adam' 또는 'adamw' 선택
use_weight_decay = True  # 가중치 감쇠 사용 여부 (True/False)
weight_decay_value = 1e-4 if use_weight_decay else 0.0  # 가중치 감쇠 값 (사용하지 않으면 0)
max_grad_norm = 5.0  # gradient clipping 최대 norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

# -----------------------------
# 2. 데이터셋 및 데이터 로더 구성
# -----------------------------
# 이미지 전처리: 학습 시 데이터 증강, 검증/테스트 시에는 단순 변환 적용
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

# 직접 촬영한 이미지들을 폴더별로 분류해두었다고 가정하고 ImageFolder를 사용
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# 데이터셋을 훈련, 검증, 테스트로 분할 (예: 70% 훈련, 15% 검증, 15% 테스트)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                        generator=torch.Generator().manual_seed(seed))

# 검증과 테스트 데이터는 transform 변경 (augmentation 없이)
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
# pretrained 모델을 불러오고, 마지막 FC 레이어를 num_classes에 맞게 설정
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
        
        # 손실 값이 NaN이거나 너무 크게 튀면 업데이트하지 않음
        if torch.isnan(loss) or loss.item() > 1e3:
            print(f"Warning: Loss spike detected at batch {batch_idx} with loss {loss.item()}. Skipping update.")
            continue
        
        loss.backward()
        # Gradient clipping을 적용하여 loss 튀는 현상을 방지
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
        # 검증 성능이 가장 좋았을 때 모델 저장
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
# 7. 저장된 모델 로드 및 테스트 수행
# -----------------------------
model.load_state_dict(torch.load(best_model_path))
true_labels, pred_labels = test(model, test_loader, device)

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

## 코드 주요 내용에 대한 설명

1. **데이터셋 구성 및 전처리**  
   - `datasets.ImageFolder`를 사용하여, 직접 촬영한 이미지를 클래스별 폴더 구조로 구성한 데이터셋을 불러옵니다.  
   - 학습 데이터에는 데이터 증강(transformations) (예, RandomHorizontalFlip)을 적용하고, 검증/테스트 데이터에는 동일한 크기 조정 및 정규화만 적용합니다.  
   - 전체 데이터셋을 70% 훈련, 15% 검증, 15% 테스트로 나누어 사용합니다.

2. **모델 생성 (timm 사용)**  
   - `timm.create_model`을 사용하여 MobileNetV4 모델을 불러오며, `num_classes`를 데이터셋에 맞게 조정합니다.  
   - pretrained 파라미터를 True로 하여 사전학습된 가중치를 사용합니다.

3. **옵티마이저 및 가중치 감쇠 옵션**  
   - 두 가지 옵티마이저 (Adam, AdamW)를 선택할 수 있으며, 변수 `optimizer_choice`를 통해 결정합니다.  
   - `use_weight_decay` 플래그에 따라 가중치 감쇠를 적용하거나 생략합니다.

4. **학습 및 검증 루프, tqdm 진행바 적용**  
   - 각 에폭마다 훈련 및 검증 루프를 진행하며, tqdm 라이브러리를 사용하여 진행상황을 실시간으로 확인할 수 있도록 하였습니다.  
   - 손실이 NaN이거나 너무 큰 경우(예: 1e3 이상의 값) 업데이트를 건너뛰도록 하여 손실이 튀는 경우에 대한 대비책을 마련하였습니다.  
   - gradient clipping(`torch.nn.utils.clip_grad_norm_`)을 적용하여 손실 값 튀는 현상을 방지합니다.

5. **Early Stopping**  
   - 검증 손실이 개선되지 않는 경우 `patience` 수 만큼 에폭을 기다린 후 학습을 중단하며, 이때 검증 성능이 가장 좋았던 시점의 모델 가중치를 저장합니다.

6. **테스트 및 Confusion Matrix**  
   - 학습 완료 후 저장된 최적 모델을 로드하여 테스트 데이터에 대해 평가한 후, sklearn의 confusion matrix 기능을 사용하여 결과를 시각화합니다.

7. **그래프 출력**  
   - 에폭마다 기록된 훈련 손실, 검증 손실 및 검증 정확도를 matplotlib을 사용하여 그래프로 그려 학습 과정을 한눈에 파악할 수 있도록 하였습니다.

이 코드는 PyTorch, timm, torchvision, tqdm, matplotlib, scikit-learn 등의 라이브러리를 사용하므로, 사용 전 필요한 라이브러리를 `pip install timm torchvision tqdm matplotlib scikit-learn` 등으로 설치해 주시기 바랍니다.

자세한 내용이나 추가 문의사항이 있으시면 언제든지 말씀해 주세요.  
  
**참고 자료:**  
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)  
- [timm GitHub 저장소](https://github.com/rwightman/pytorch-image-models)  
- [tqdm GitHub](https://github.com/tqdm/tqdm)
