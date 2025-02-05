JSON 형식의 데이터셋을 처리하는 PyTorch `Dataset` 클래스를 구현하고, `DataLoader`를 구성하는 방법을 설명하겠습니다.

---

## 1. `Dataset` 클래스 구현
JSON 파일을 읽고, 이미지와 텍스트를 함께 로드하는 `MobileCLIPDataset` 클래스를 정의합니다.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from transformers import CLIPTokenizer

class MobileCLIPDataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, transform=None):
        """
        MobileCLIP을 위한 데이터셋 클래스

        :param json_file: 이미지-텍스트 쌍이 저장된 JSON 파일 경로
        :param image_dir: 이미지 파일이 저장된 폴더 경로
        :param tokenizer: 텍스트 데이터를 토큰화할 CLIP 토크나이저
        :param transform: 이미지에 적용할 변환(transform)
        """
        # JSON 파일 로드
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # MobileCLIP은 224x224 해상도를 사용
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 로드 및 변환
        img_path = os.path.join(self.image_dir, self.data[idx]["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 텍스트 로드 및 토큰화
        text = self.data[idx]["text"]
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

        return image, text_tokens['input_ids'].squeeze(0), text_tokens['attention_mask'].squeeze(0)
```

---

## 2. `DataLoader` 생성
위에서 정의한 `MobileCLIPDataset`을 이용하여 `DataLoader`를 생성합니다.

```python
# CLIP용 토크나이저 로드
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 데이터셋 생성
dataset = MobileCLIPDataset(json_file="dataset.json", image_dir="images", tokenizer=tokenizer)

# 데이터 로더 생성
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 데이터 로더 테스트
for batch in data_loader:
    images, text_ids, attention_masks = batch
    print("Image batch shape:", images.shape)
    print("Text token batch shape:", text_ids.shape)
    print("Attention mask shape:", attention_masks.shape)
    break  # 한 배치만 확인
```

---

## 3. 주요 설명
1. **JSON 데이터 로드**
   - `json.load(f)`를 이용해 리스트 형식의 JSON을 읽음
   - 각 항목에서 `"image"`와 `"text"` 필드를 추출

2. **이미지 전처리**
   - 이미지는 `224x224` 크기로 변환
   - 정규화 `(0.5, 0.5, 0.5)`로 설정하여 MobileCLIP 학습 환경과 맞춤

3. **텍스트 토큰화**
   - `CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")`를 이용해 토큰화
   - `padding='max_length'`, `truncation=True` 설정

4. **`DataLoader` 설정**
   - `batch_size=32`: 한 번에 32개 샘플 로드
   - `shuffle=True`: 데이터를 랜덤하게 섞음
   - `num_workers=4`: 데이터 로딩 속도를 높이기 위해 4개 프로세스 사용

---

## 4. 실행 예시 출력
```
Image batch shape: torch.Size([32, 3, 224, 224])
Text token batch shape: torch.Size([32, 77])
Attention mask shape: torch.Size([32, 77])
```

이제 MobileCLIP을 학습할 때 이 `DataLoader`를 활용하면 됩니다. 🚀
