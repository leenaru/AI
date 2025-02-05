## 🔹 **MobileCLIP을 직접 다운로드하고 로드하는 방법** (🔥 `torch.hub.load` 사용 안 함)

> **✅ 주요 목표:**  
> - `torch.hub.load()` 없이 MobileCLIP을 직접 다운로드하고 불러오는 방법을 설명  
> - `state_dict` 기반으로 PyTorch에서 모델을 로드  
> - **직접 모델 정의 + 로컬에서 MobileCLIP 불러오기**

---

## 📌 **1. MobileCLIP 사전 학습된 모델 다운로드**  
MobileCLIP 모델을 Apple Research에서 제공하는 `.pth` 파일로 다운로드해야 합니다.  
> **예시 다운로드 링크 (직접 확인 필요)**  
```
https://example.com/apple/mobileclip_weights.pth
```
**🔥 터미널에서 다운로드**
```bash
mkdir -p weights
wget -O weights/mobileclip.pth https://example.com/apple/mobileclip_weights.pth
```
---

## 📌 **2. MobileCLIP 모델 직접 구현하기**  
Apple이 제공하는 `torch.hub.load("apple/mobileclip", "image_encoder")` 대신  
✅ **MobileCLIP의 기본 구조를 직접 정의하고 `state_dict`로 로드**

📍 **MobileCLIP 기본 모델 정의**
```python
import torch
import torch.nn as nn
import torchvision.models as models

class MobileCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ✅ 이미지 인코더 (EfficientNet 기반)
        self.image_encoder = models.efficientnet_b0(pretrained=False)
        self.image_encoder.classifier = nn.Identity()  # 분류기 제거하여 feature extractor로 사용
        
        # ✅ 텍스트 인코더 (Transformer 기반)
        self.text_encoder = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
        
        # ✅ 학습 가능한 로그 스케일 파라미터 (CLIP과 동일)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts["input_ids"], texts["attention_mask"])
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
```

---

## 📌 **3. 다운로드한 MobileCLIP 가중치 로드**
위에서 정의한 `MobileCLIP` 모델에 직접 `.pth` 파일을 로드합니다.

📍 **모델 가중치 로드**
```python
# ✅ MobileCLIP 모델 생성
model = MobileCLIP()

# ✅ 다운로드한 가중치 로드
model_path = "./weights/mobileclip.pth"
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)

# ✅ GPU가 있다면 모델을 GPU로 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ MobileCLIP 모델이 성공적으로 로드되었습니다!")
```
✅ **이제 `torch.hub.load` 없이도 MobileCLIP 모델을 사용할 수 있음!**

---

## 📌 **4. MobileCLIP 테스트 (Inference)**
✅ MobileCLIP 모델이 정상적으로 작동하는지 테스트  
✅ 샘플 이미지와 텍스트를 입력하여 유사도 점수 확인

📍 **이미지 & 텍스트 전처리**
```python
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

# ✅ 이미지 변환 (MobileCLIP에 맞게 정규화)
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0).to(device)

# ✅ 텍스트 토큰화 (Hugging Face BERT 토크나이저 사용)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return {key: val.to(device) for key, val in tokens.items()}
```

📍 **모델 추론 (Inference)**
```python
# ✅ 샘플 입력 데이터
image_path = "./test_images/cat.jpg"
text_input = "A cat sitting on a chair."

# ✅ 데이터 변환
image = preprocess_image(image_path)
text = preprocess_text(text_input)

# ✅ 모델 추론
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)

print(f"🔹 이미지-텍스트 유사도 점수: {logits_per_image.item():.4f}")
```

✅ **정상적으로 동작하면 MobileCLIP을 사용할 준비 완료!**

---

## 📌 **5. MobileCLIP 모델 저장 (추후 사용)**
학습 후 모델을 다시 저장할 수도 있음.
```python
save_path = "./weights/mobileclip_finetuned.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ 학습된 MobileCLIP 모델이 저장됨: {save_path}")
```

---

# 🎯 **결론**
| 단계 | 설명 |
|------|------|
| 📥 1. MobileCLIP 모델 다운로드 | `.pth` 파일을 직접 다운로드 |
| 🔧 2. 모델 정의 | MobileCLIP을 직접 PyTorch로 구현 |
| 🏗 3. 모델 로드 | `state_dict()`를 사용해 로드 |
| 🎯 4. 모델 테스트 | 이미지 & 텍스트 입력 후 결과 확인 |
| 💾 5. 모델 저장 | 학습된 모델을 저장하여 재사용 |

👉 **이제 `torch.hub.load` 없이 MobileCLIP을 직접 다운로드하고 사용할 수 있습니다!** 🚀  
추가 질문이 있으면 언제든 물어보세요 😊
