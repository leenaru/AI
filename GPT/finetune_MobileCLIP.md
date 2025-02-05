직접 촬영한 이미지를 MobileCLIP의 파인튜닝 데이터로 사용하려면 **폴더 구조, JSON 파일 형식, 데이터 로드 방식**을 체계적으로 정리해야 합니다. 아래는 MobileCLIP 학습에 적합한 폴더 및 파일 구조, JSON 예제, 그리고 데이터 로드 방식을 설명합니다.

---

## **1. 폴더 구조**
데이터셋을 체계적으로 관리하기 위해 다음과 같은 폴더 구조를 권장합니다.

```
/mobileclip_dataset/
│── /images/             # 촬영한 이미지들이 저장되는 폴더
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.jpg
│   ├── ...
│── /annotations/         # JSON 파일 저장 폴더
│   ├── dataset.json
│── train.py              # MobileCLIP 학습 스크립트
```

- `/images/` 폴더:  
  - 직접 촬영한 이미지들을 저장합니다.  
  - 파일명은 `image_001.jpg`, `image_002.jpg` 등으로 유지하는 것이 좋습니다.
  - 같은 이미지에 대해 여러 개의 텍스트 설명이 존재할 수도 있습니다.  

- `/annotations/dataset.json`:  
  - 이미지-텍스트 매칭 정보를 저장하는 JSON 파일입니다.  
  - 각 이미지에 대한 설명(캡션)을 포함해야 합니다.

---

## **2. JSON 파일 예제 (`dataset.json`)**
JSON 파일은 이미지 파일명과 텍스트 설명을 매칭하는 형태로 저장됩니다.

```json
[
    {
        "image": "image_001.jpg",
        "text": "A small white cat sitting on a wooden table."
    },
    {
        "image": "image_002.jpg",
        "text": "A red sports car parked in front of a modern building."
    },
    {
        "image": "image_003.jpg",
        "text": "A beautiful sunset over the ocean with seagulls flying."
    }
]
```

- `image`: 이미지 파일 이름 (확장자 포함)  
- `text`: 해당 이미지의 설명 (영어 문장을 권장)  

> **✅ 팁**: 하나의 이미지에 대해 여러 개의 캡션을 저장하려면 JSON을 다음과 같이 수정할 수도 있습니다.
```json
[
    {
        "image": "image_001.jpg",
        "text": ["A white cat sitting on a wooden table.", "A cute cat resting indoors."]
    }
]
```
이 경우, 학습 데이터 로더에서 여러 개의 설명을 랜덤으로 선택할 수 있도록 처리하면 됩니다.

---

## **3. 데이터 로더 코드 (`dataset.json` 로드)**
JSON 파일을 읽어 `Dataset` 클래스로 변환하고, MobileCLIP 학습에 사용할 수 있도록 합니다.

### **(1) 데이터 로드 코드**
```python
import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor

# JSON 데이터셋 파일 경로
JSON_FILE_PATH = "mobileclip_dataset/annotations/dataset.json"
IMAGE_FOLDER_PATH = "mobileclip_dataset/images"

# CLIP 모델의 데이터 처리기 로드
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# JSON 파일 로드
with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# PyTorch Dataset 클래스 정의
class MobileCLIPDataset(Dataset):
    def __init__(self, dataset, processor, image_folder):
        self.dataset = dataset
        self.processor = processor
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = os.path.join(self.image_folder, item["image"])
        text = item["text"]

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)

        return {key: val.squeeze(0) for key, val in inputs.items()}

# 데이터 로더 생성
dataset = MobileCLIPDataset(dataset, processor, IMAGE_FOLDER_PATH)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 데이터 확인
sample = next(iter(dataloader))
print(sample["pixel_values"].shape, sample["input_ids"].shape)  # (batch_size, 3, H, W), (batch_size, seq_len)
```

### **(2) 설명**
- JSON을 로드하여 **이미지-텍스트 쌍**을 `Dataset` 형식으로 변환합니다.
- `CLIPProcessor`를 이용해 **이미지를 텐서로 변환**합니다.
- `DataLoader`를 사용하여 학습할 데이터 배치를 생성합니다.

---

## **4. MobileCLIP 학습 실행**
위에서 만든 `dataloader`를 이용하여 MobileCLIP 모델을 학습할 수 있습니다.

```python
import torch
import torch.optim as optim
from transformers import CLIPModel

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

# 손실 함수 및 옵티마이저
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)

# 학습 루프
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)

        # 텍스트와 이미지 간 대조 학습(Contrastive Loss)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        labels = torch.arange(len(logits_per_image)).to(device)

        loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
```

---

## **5. MobileCLIP 학습 후 모델 저장 및 변환**
학습한 모델을 저장하고 모바일 환경에서 사용할 수 있도록 변환하는 과정입니다.

### **(1) 학습된 모델 저장**
```python
model.save_pretrained("mobileclip_finetuned")
processor.save_pretrained("mobileclip_finetuned")
```

### **(2) ONNX 변환 (모바일 환경 지원)**
```python
import torch.onnx

dummy_input = {key: torch.randn(1, *val.shape[1:]).to(device) for key, val in next(iter(dataloader)).items()}
torch.onnx.export(model, (dummy_input["input_ids"], dummy_input["pixel_values"]), "mobileclip.onnx")
```

### **(3) CoreML 변환 (iOS 앱용)**
```python
import coremltools as ct

coreml_model = ct.convert(model, inputs=[ct.ImageType()])
coreml_model.save("MobileCLIP.mlmodel")
```

---

## **요약**
1. **폴더 구조 정리**  
   - `/images/` 폴더에 촬영한 이미지 저장  
   - `/annotations/dataset.json`에 이미지-텍스트 매칭 정보 저장  

2. **JSON 파일 작성**  
   - 각 이미지에 대한 설명을 포함하는 JSON 생성  

3. **데이터 로드**  
   - `MobileCLIPDataset` 클래스를 사용해 JSON을 `DataLoader`로 변환  

4. **MobileCLIP 학습**  
   - 모델을 로드하고 학습을 진행  

5. **모델 저장 및 변환**  
   - 학습된 모델을 저장하고 ONNX 또는 CoreML로 변환  

이제 직접 촬영한 이미지를 활용해 MobileCLIP을 파인튜닝할 수 있습니다!
