### 📌 요약: **MobileCLIP을 내 사진으로 파인튜닝하는 전체 과정**
1. **데이터셋 준비**: 직접 촬영한 사진을 모으고, 적절한 텍스트 설명을 작성.
2. **데이터 라벨링**: 사진과 텍스트 매칭을 JSON 또는 CSV 형식으로 정리.
3. **데이터 전처리**: MobileCLIP 모델에 맞게 이미지와 텍스트 변환.
4. **MobileCLIP 모델 설정**: Apple의 MobileCLIP을 가져와 특정 레이어만 학습.
5. **파인튜닝**: 직접 만든 데이터셋으로 학습 진행.
6. **성능 평가 및 최적화**: 모델이 잘 학습되었는지 확인하고 튜닝.

---

## 🔹 1. 데이터셋 준비: 직접 찍은 사진을 사용하기

> **✅ 목표**: 내가 찍은 사진을 MobileCLIP 모델이 이해하도록 학습시키기.

### 📌 데이터 수집
- **직접 촬영한 사진** 📸: iPhone, Android, DSLR 등으로 원하는 사진을 촬영.
- **사진 종류**:
  - 인물, 동물, 사물, 풍경, 제품 등 원하는 카테고리별로 다양하게 촬영.
  - 같은 객체라도 다양한 각도와 조명에서 촬영.
  - 예제: "고양이", "책상 위의 커피잔", "서울 야경"

---

## 🔹 2. 데이터 라벨링: 사진과 텍스트 연결

> **✅ 목표**: 사진과 설명을 텍스트 데이터로 매칭하기.

### 📌 CSV 파일로 만들기 (추천)
각 사진에 대한 설명을 CSV 파일로 정리하면 관리하기 편리함.

```csv
image_path,text
./images/cat1.jpg,고양이가 소파 위에서 졸고 있다.
./images/cat2.jpg,갈색 고양이가 창밖을 보고 있다.
./images/coffee.jpg,흰색 머그컵에 커피가 가득 차 있다.
```

> **✔️ CSV 저장 방법 (Python 코드)**
```python
import csv

data = [
    ("./images/cat1.jpg", "고양이가 소파 위에서 졸고 있다."),
    ("./images/cat2.jpg", "갈색 고양이가 창밖을 보고 있다."),
    ("./images/coffee.jpg", "흰색 머그컵에 커피가 가득 차 있다."),
]

with open("dataset.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "text"])
    writer.writerows(data)
```

---

## 🔹 3. 데이터 전처리: MobileCLIP 입력 형식으로 변환

> **✅ 목표**: MobileCLIP 모델이 이해할 수 있도록 이미지와 텍스트를 변환.

### 📌 ① 이미지 전처리 (크기 조정 & 정규화)
MobileCLIP 모델은 `224x224` 크기의 이미지를 입력으로 받음.

```python
import torchvision.transforms as T
from PIL import Image

image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image)
```

---

### 📌 ② 텍스트 토큰화
MobileCLIP의 텍스트 인코더는 일반적인 BERT 기반 토크나이저 사용.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True)
```

---

## 🔹 4. MobileCLIP 모델 설정

> **✅ 목표**: Apple MobileCLIP 모델을 불러와 일부 레이어만 학습.

```python
import torch
import torch.nn as nn

class MobileCLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text["input_ids"], text["attention_mask"])
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

# MobileCLIP 모델 로드
image_encoder = torch.hub.load("apple/mobileclip", "image_encoder")
text_encoder = torch.hub.load("apple/mobileclip", "text_encoder")

model = MobileCLIPModel(image_encoder, text_encoder)
```

---

## 🔹 5. MobileCLIP 모델 파인튜닝

> **✅ 목표**: 직접 찍은 사진 데이터셋으로 모델 학습.

```python
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# 데이터 불러오기
import pandas as pd

dataset = pd.read_csv("dataset.csv")

# 이미지 및 텍스트 변환
images, texts = [], []
for _, row in dataset.iterrows():
    images.append(preprocess_image(row["image_path"]))
    texts.append(preprocess_text(row["text"]))

images = torch.stack(images)
texts = {key: torch.cat([t[key] for t in texts], dim=0) for key in texts[0]}

# 옵티마이저 및 손실 함수
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

# 학습 루프
for epoch in range(5):
    model.train()
    optimizer.zero_grad()

    logits_per_image, logits_per_text = model(images, texts)
    labels = torch.arange(len(images)).to(logits_per_image.device)

    loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2
    print(f"Epoch {epoch}, Loss: {loss.item()}")

    loss.backward()
    optimizer.step()
```

---

## 🔹 6. 모델 평가 및 성능 튜닝

> **✅ 목표**: MobileCLIP이 학습한 결과를 평가.

```python
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch
            logits_per_image, _ = model(images, texts)
            preds = logits_per_image.argmax(dim=1)
            correct += (preds == torch.arange(len(images)).to(preds.device)).sum().item()
            total += len(images)
    return correct / total
```

---

## 🚀 **결론: 내가 찍은 사진을 MobileCLIP으로 학습하는 전체 과정**
| 단계 | 설명 |
|------|------|
| 📸 1. 데이터셋 준비 | 직접 사진 촬영 후 저장 |
| 📝 2. 데이터 라벨링 | CSV로 사진과 텍스트 연결 |
| 🛠 3. 데이터 전처리 | 이미지 변환 및 텍스트 토큰화 |
| 🏗 4. MobileCLIP 모델 설정 | Apple MobileCLIP 로드 |
| 🎯 5. 파인튜닝 | 직접 찍은 사진으로 모델 학습 |
| 📊 6. 성능 평가 | 모델이 잘 학습했는지 확인 |

👉 **이 과정을 따라 하면 MobileCLIP이 내 사진을 학습할 수 있습니다!**  
추가로 궁금한 점 있으면 질문 주세요 😊
