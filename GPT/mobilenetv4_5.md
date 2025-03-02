아래는 stratified split과 k-fold cross validation 옵션을 선택하여 사용할 수 있도록 작성한 전체 PyTorch 코드 예제입니다.  
코드 최상단의 변수 `split_method` 값을 `"random"`, `"stratified"`, 또는 `"kfold"`로 변경하여 데이터셋 분할 방식을 선택할 수 있습니다.  
이 코드는 직접 촬영한 이미지들을 ImageFolder 형식으로 불러와서  
- 기본 random split (70% train / 15% validation / 15% test),  
- 클래스 분포를 유지하는 stratified split, 또는  
- k-fold cross validation (먼저 test set을 분리한 후 train/validation에 대해 k-fold CV 수행 후 최종 모델을 재학습)  
방식 중 하나로 학습을 진행하며, tqdm 진행바, early stopping, gradient clipping, 옵티마이저(Adam/AdamW), 가중치 감쇠 옵션, 테스트 시 Accuracy, Precision, Recall, F1 및 소요시간 측정, Confusion Matrix 시각화, 에폭별 학습 곡선 출력 등을 포함합니다.

---

```python
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
import timm  # timm 라이브러리로 모델 생성
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# -----------------------------
# 1. 하이퍼파라미터 및 옵션 설정
# -----------------------------
data_dir = './data'  # 직접 촬영한 이미지들이 클래스별 폴더로 정리된 경로 (예: data/class1, data/class2, ...)
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
patience = 5  # 검증 성능 개선 없을 때 기다릴 에폭 수
optimizer_choice = 'adam'  # 'adam' 또는 'adamw'
use_weight_decay = True         # 가중치 감쇠 사용 여부 (True/False)
weight_decay_value = 1e-4 if use_weight_decay else 0.0
max_grad_norm = 5.0             # gradient clipping 최대 norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

# -----------------------------
# 2. 분할 방식 선택 옵션
# -----------------------------
# split_method 옵션: "random" (기본 랜덤 분할), "stratified" (클래스 분포 유지 분할), "kfold" (k-fold cross validation)
split_method = "random"  # "random", "stratified", "kfold" 중 선택
k_folds = 5  # split_method가 "kfold"일 때 사용할 fold 수

# -----------------------------
# 3. 데이터 전처리 (transforms) 설정
# -----------------------------
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

# -----------------------------
# 4. 데이터셋 로드 (ImageFolder)
# -----------------------------
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# -----------------------------
# 5. 학습/검증/테스트 함수 정의
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Train Epoch {epoch}")
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 손실 값이 NaN이거나 너무 크면 업데이트 건너뛰기
        if torch.isnan(loss) or loss.item() > 1e3:
            print(f"Warning: Loss spike at batch {batch_idx} (loss {loss.item():.4f}). Skipping update.")
            continue
        
        loss.backward()
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
# 6. 분할 방식에 따른 데이터셋 구성 및 학습/평가 실행
# -----------------------------
if split_method in ["random", "stratified"]:
    # -- random 또는 stratified split --
    if split_method == "random":
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    elif split_method == "stratified":
        indices = list(range(len(full_dataset)))
        targets = [full_dataset.samples[i][1] for i in indices]
        # 먼저 70% train, 30% temp로 분할 (stratify 적용)
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, stratify=targets, random_state=seed
        )
        temp_targets = [targets[i] for i in temp_indices]
        # temp를 50:50으로 나누어 validation (15%)와 test (15%)로 분할
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, stratify=temp_targets, random_state=seed
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
    
    # 검증/테스트 시에는 test_transform 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성 (timm의 MobileNetV4)
    num_classes = len(full_dataset.classes)
    model = timm.create_model('mobilenetv4_100', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
    elif optimizer_choice.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
    else:
        raise ValueError("optimizer_choice must be 'adam' or 'adamw'")
    
    # 학습 루프 (Early Stopping 적용)
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epoch(s).")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print(f"\nTraining finished. Best validation loss at epoch {best_epoch}.")
    
    # 테스트 수행 및 평가 metric 측정
    model.load_state_dict(torch.load(best_model_path))
    start_time = time.time()
    true_labels, pred_labels = test(model, test_loader, device)
    elapsed_time = time.time() - start_time
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
    
    # Confusion Matrix 시각화
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix on Test Data")
    plt.show()
    
    # 에폭별 학습 손실 및 정확도 그래프 출력
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

else:  # split_method == "kfold"
    # -- k-fold cross validation 적용 --
    # 먼저 전체 데이터셋에서 stratified 방식으로 test set (15%) 분리
    indices = list(range(len(full_dataset)))
    targets = [full_dataset.samples[i][1] for i in indices]
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.15, stratify=targets, random_state=seed
    )
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = test_transform
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # train_val 데이터에 대해 k-fold cross validation (StratifiedKFold)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, [targets[i] for i in train_val_indices])):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        fold_train_indices = [train_val_indices[i] for i in train_idx]
        fold_val_indices = [train_val_indices[i] for i in val_idx]
        train_dataset_fold = Subset(full_dataset, fold_train_indices)
        val_dataset_fold = Subset(full_dataset, fold_val_indices)
        train_dataset_fold.dataset.transform = train_transform
        val_dataset_fold.dataset.transform = test_transform
        
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 모델 및 옵티마이저 초기화 (매 fold마다 새로 생성)
        num_classes = len(full_dataset.classes)
        model = timm.create_model('mobilenetv4_100', pretrained=True, num_classes=num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        if optimizer_choice.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
        elif optimizer_choice.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
        else:
            raise ValueError("optimizer_choice must be 'adam' or 'adamw'")
        
        best_val_loss_fold = float('inf')
        early_stop_counter = 0
        fold_best_model_path = f"best_model_fold{fold}.pth"
        
        for epoch in range(1, num_epochs+1):
            print(f"Fold {fold+1} - Epoch {epoch}/{num_epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss_fold:
                best_val_loss_fold = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), fold_best_model_path)
                print("Best model for this fold saved.")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epoch(s) in this fold.")
            if early_stop_counter >= patience:
                print("Early stopping triggered for this fold.")
                break
        
        # fold별 검증 성능 평가
        model.load_state_dict(torch.load(fold_best_model_path))
        true_labels_fold, pred_labels_fold = test(model, val_loader, device)
        fold_accuracy = accuracy_score(true_labels_fold, pred_labels_fold)
        fold_metrics.append(fold_accuracy)
        print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")
    
    avg_cv_accuracy = np.mean(fold_metrics)
    print("\n--- K-Fold Cross Validation Results ---")
    for i, acc in enumerate(fold_metrics):
        print(f"Fold {i+1} Accuracy: {acc:.4f}")
    print(f"Average K-Fold CV Accuracy: {avg_cv_accuracy:.4f}")
    
    # 최종 모델 재학습: train_val 전체 데이터를 대상으로 stratified split (예: 90% train / 10% validation)
    train_val_targets = [targets[i] for i in train_val_indices]
    final_train_idx, final_val_idx = train_test_split(
        train_val_indices, test_size=0.1, stratify=train_val_targets, random_state=seed
    )
    train_dataset_final = Subset(full_dataset, final_train_idx)
    val_dataset_final = Subset(full_dataset, final_val_idx)
    train_dataset_final.dataset.transform = train_transform
    val_dataset_final.dataset.transform = test_transform
    
    train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = timm.create_model('mobilenetv4_100', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
    elif optimizer_choice.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
    else:
        raise ValueError("optimizer_choice must be 'adam' or 'adamw'")
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = "best_model_final.pth"
    final_train_losses = []
    final_val_losses = []
    final_val_accuracies = []
    
    for epoch in range(1, num_epochs+1):
        print(f"\nFinal Model Training - Epoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        final_train_losses.append(train_loss)
        final_val_losses.append(val_loss)
        final_val_accuracies.append(val_acc)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best final model saved.")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epoch(s).")
        if early_stop_counter >= patience:
            print("Early stopping triggered for final model training.")
            break
    
    print("Final model training finished.")
    
    # 테스트 평가 (최종 모델)
    model.load_state_dict(torch.load(best_model_path))
    start_time = time.time()
    true_labels, pred_labels = test(model, test_loader, device)
    elapsed_time = time.time() - start_time
    test_accuracy  = accuracy_score(true_labels, pred_labels)
    test_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    test_recall    = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    test_f1        = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    print("\n--- Final Test Metrics ---")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall   : {test_recall:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print(f"Elapsed Time  : {elapsed_time:.2f} seconds")
    
    # Confusion Matrix 시각화
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix on Test Data")
    plt.show()
    
    # 최종 모델 학습 곡선 그래프 출력
    epochs_range = range(1, len(final_train_losses)+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, final_train_losses, label='Train Loss')
    plt.plot(epochs_range, final_val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch (Final Model)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, final_val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy per Epoch (Final Model)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nK-Fold Cross Validation Accuracies:")
    for i, acc in enumerate(fold_metrics):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Average K-Fold CV Accuracy: {avg_cv_accuracy:.4f}")
```

---

### 코드 주요 내용 요약

1. **데이터셋 분할 옵션:**  
   - `split_method`에 따라 **random split**, **stratified split** 또는 **k-fold cross validation** 방식을 선택합니다.  
   - stratified split은 sklearn의 `train_test_split`을 사용하여 클래스 분포를 유지하며 train/val/test로 분할합니다.  
   - k-fold의 경우 먼저 전체 데이터에서 test set(15%)을 분리한 후, 나머지 데이터를 StratifiedKFold로 나누어 각 fold별로 학습 및 검증을 수행하고 최종 모델을 재학습합니다.

2. **모델 및 학습:**  
   - timm 라이브러리의 MobileNetV4를 사용하며, 옵티마이저(Adam/AdamW)와 가중치 감쇠, gradient clipping, early stopping을 적용합니다.
   
3. **평가 및 시각화:**  
   - 테스트 후 Accuracy, Precision, Recall, F1, elapsed time을 출력하고, Confusion Matrix와 에폭별 학습 곡선을 matplotlib로 시각화합니다.

원하는 옵션에 따라 `split_method`와 `k_folds` 값을 변경하여 사용하시면 됩니다.  
