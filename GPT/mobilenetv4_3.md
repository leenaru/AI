요구사항을 모두 반영한 PyTorch 코드를 작성했습니다. MobileNetV4는 아직 공식적으로 출시되지 않았기 때문에, 대신 MobileNetV3를 사용했습니다. 코드는 다음과 같습니다:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataset(user_dir='user_images', val_size=0.2, test_size=0.1):
    image_paths = []
    labels = []
    class_names = os.listdir(user_dir)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(user_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(class_idx)
    
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels)
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=val_size, stratify=train_val_labels)
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels, class_names

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score = self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return history

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main(optimizer_type='adam', use_weight_decay=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_images, val_images, test_images, train_labels, val_labels, test_labels, class_names = create_dataset()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(train_images, train_labels, transform)
    val_dataset = CustomDataset(val_images, val_labels, transform)
    test_dataset = CustomDataset(test_images, test_labels, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(class_names)
    model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01 if use_weight_decay else 0)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01 if use_weight_decay else 0)
    else:
        raise ValueError("Invalid optimizer type. Choose 'adam' or 'adamw'.")
    
    early_stopping = EarlyStopping(patience=7, verbose=True)
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device, early_stopping=early_stopping)
    
    plot_history(history)
    
    model.load_state_dict(torch.load('checkpoint.pth'))
    evaluate_model(model, test_loader, device, class_names)

if __name__ == "__main__":
    main(optimizer_type='adam', use_weight_decay=True)
```

이 코드는 다음과 같은 기능을 포함합니다:

1. 사용자 정의 데이터셋 생성 및 사용
2. tqdm을 사용한 진행 상황 표시
3. 훈련, 검증, 테스트 데이터 분할
4. 에폭마다 검증 성능 모니터링
5. Early stopping 적용
6. 에폭당 손실과 정확도 그래프 생성
7. Adam과 AdamW 옵티마이저 선택 가능
8. 가중치 감쇠 사용 여부 선택 가능
9. 손실 튀는 경우 대비 (Early stopping 사용)
10. timm을 사용한 MobileNetV3 모델 생성

사용 방법:
1. 'user_images' 폴더에 클래스별로 이미지를 저장합니다.
2. 코드 실행 시 자동으로 데이터셋 생성, 모델 학습, 평가가 진행됩니다.
3. 옵티마이저와 가중치 감쇠 사용 여부는 main 함수 호출 시 인자로 지정할 수 있습니다.

주의사항:
- timm 라이브러리 설치가 필요합니다: `pip install timm`
- 충분한 이미지 데이터를 확보해야 합니다 (클래스당 최소 50장 이상 권장)
- GPU 사용 시 학습 속도가 크게 향상됩니다.

이 코드는 PyTorch와 timm을 사용하여 이미지 분류 작업을 수행하며, 사용자 정의 데이터셋에 대해 MobileNetV3 모델을 학습하고 평가합니다.

---
Perplexity로부터의 답변: pplx.ai/share
