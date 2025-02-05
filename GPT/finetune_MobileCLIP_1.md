### **요약**
이미지 분류용 데이터셋을 만들어 **MobileCLIP 모델**을 **PyTorch**에서 파인튜닝하는 과정을 단계별로 설명합니다.

1. **데이터셋 준비**
   - `ImageFolder` 형식으로 데이터 구성
   - `train`, `val`, `test` 폴더 분할

2. **데이터 로드 및 전처리**
   - PyTorch `torchvision.datasets.ImageFolder` 활용
   - `transforms`로 데이터 증강

3. **MobileCLIP 모델 불러오기**
   - `open_clip` 라이브러리 활용

4. **모델 파인튜닝**
   - 분류기(Classifier) 레이어 수정
   - 손실 함수 및 최적화 설정

5. **모델 학습**
   - GPU 활용하여 학습 진행
   - 학습 과정 시각화

6. **모델 평가**
   - 테스트 데이터로 정확도 측정
   - 일부 예측 결과 확인

---

## **1. 데이터셋 준비**
먼저, 이미지 분류용 데이터셋을 준비합니다.

### **(1) 폴더 구조**
아래처럼 폴더를 구성하면 PyTorch의 `ImageFolder` 클래스로 쉽게 불러올 수 있습니다.

```
dataset/
 ├── train/
 │   ├── class1/  (예: 고양이)
 │   │   ├── image1.jpg
 │   │   ├── image2.jpg
 │   ├── class2/  (예: 개)
 │   │   ├── image3.jpg
 │   │   ├── image4.jpg
 ├── val/
 │   ├── class1/
 │   ├── class2/
 ├── test/
 │   ├── class1/
 │   ├── class2/
```

### **(2) 데이터셋 다운로드**
공개 데이터셋을 다운로드하여 사용할 수도 있습니다.

```python
import os
import torchvision.datasets as datasets

dataset_dir = "dataset/"
os.makedirs(dataset_dir, exist_ok=True)

# 샘플 데이터셋 다운로드 (예: CIFAR-10)
datasets.CIFAR10(root=dataset_dir, train=True, download=True)
```

---

## **2. 데이터 로드 및 전처리**
### **(1) 데이터 변환(Augmentation)**
MobileCLIP은 CLIP 모델 기반이므로, **CLIP의 기본 변환 방식**을 따라야 합니다.

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# CLIP 모델에 적합한 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileCLIP은 224x224 사용
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
])

# 데이터 로드
train_dataset = ImageFolder(root="dataset/train", transform=transform)
val_dataset = ImageFolder(root="dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 클래스 이름 확인
print("클래스 목록:", train_dataset.classes)
```

---

## **3. MobileCLIP 모델 불러오기**
MobileCLIP은 **`open_clip` 라이브러리**를 사용하여 로드할 수 있습니다.

```python
import open_clip

# MobileCLIP 모델 불러오기
model_name = "mobilevit_xxs"
pretrained = "laion5B"

model, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# 모델 구조 확인
print(model)
```

---

## **4. 모델 파인튜닝**
MobileCLIP은 기본적으로 이미지-텍스트 매칭을 수행하는 모델이지만, **분류 문제(Classification)**에 맞게 수정해야 합니다.

### **(1) 분류기(Classifier) 추가**
MobileCLIP의 **출력 임베딩**을 가져와 **새로운 분류기**를 추가합니다.

```python
import torch.nn as nn

# MobileCLIP의 출력을 받아 분류 레이어 추가
class MobileCLIPClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(MobileCLIPClassifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(640, num_classes)  # MobileViT XXS는 640차원 출력

    def forward(self, x):
        x = self.base_model.encode_image(x)
        x = self.fc(x)
        return x

# 클래스 개수 (예: 2개)
num_classes = len(train_dataset.classes)

# 모델 생성
model = MobileCLIPClassifier(model, num_classes).cuda()

# 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## **5. 모델 학습**
모델을 훈련하고 검증합니다.

```python
import torch

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 검증
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
```

---

## **6. 모델 평가 및 예측**
### **(1) 테스트 데이터 평가**
```python
test_dataset = ImageFolder(root="dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

### **(2) 샘플 예측 결과 확인**
```python
import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터 로드
sample_images, sample_labels = next(iter(test_loader))
sample_images, sample_labels = sample_images.cuda(), sample_labels.cuda()

# 예측 수행
outputs = model(sample_images)
_, predicted = torch.max(outputs, 1)

# 결과 시각화
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()

for i in range(16):
    img = sample_images[i].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize
    label = train_dataset.classes[predicted[i]]

    axes[i].imshow(img)
    axes[i].set_title(label)
    axes[i].axis("off")

plt.show()
```

---

## **결론**
1. `ImageFolder`로 이미지 분류용 데이터셋을 생성
2. `open_clip`으로 **MobileCLIP 모델 로드**
3. **새로운 분류기**를 추가하여 파인튜닝
4. GPU를 활용하여 학습 및 검증 수행
5. 모델 평가 후 샘플 예측 결과 확인

이제 MobileCLIP을 활용한 이미지 분류 모델을 만들고, 직접 데이터를 추가해 실험해볼 수 있습니다! 추가 질문이 있다면 말씀해주세요.
