아래는 **MobileCLIP에 LoRA를 적용하고 Contrastive Loss를 수동으로 계산하는 전체 코드**입니다.  
또한, **MobileCLIP의 Text Encoder는 미리 전체 클래스를 받아 임베딩을 해야 하므로**, 이를 반영하여 **Text Encoder를 사전에 임베딩하고 재사용하는 방식**으로 수정했습니다.

---

### **🔥 MobileCLIP + LoRA + Contrastive Loss 적용 코드 (Text Encoder 사전 임베딩 포함)**
```python
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType

# 1. MobileCLIP 모델 로드
model_name = "apple/ml-mobileclip"  # 실제 사용 가능한 MobileCLIP 모델로 변경 필요
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. LoRA 설정 및 적용
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # LoRA 랭크
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["text_model.encoder.layers", "vision_model.encoder.layers"],  # 적용 대상
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 전체 클래스 텍스트 임베딩 (Text Encoder 사전 계산)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class_labels = ["a cat", "a dog", "a car", "a tree"]  # 실제 사용하려는 클래스 리스트
text_inputs = processor(text=class_labels, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_outputs = model.get_text_features(**text_inputs)  # text_encoder를 사전에 처리
    text_features = F.normalize(text_outputs, dim=-1)  # 정규화

print("Text features shape:", text_features.shape)  # (num_classes, embedding_dim)

# 4. 데이터셋 및 데이터로더 정의 (사용자 데이터셋에 맞게 수정)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, processor, num_samples=100):
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)  # 가짜 이미지 데이터 (실제 이미지로 교체 필요)
        label = torch.randint(0, len(class_labels), (1,)).item()  # 랜덤 레이블 할당
        return {"image": image, "label": label}

train_dataset = DummyDataset(processor)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 5. 옵티마이저 및 학습 설정
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# 6. Contrastive Loss를 수동 계산하며 모델 학습
for epoch in range(3):  # 3 Epoch 예제
    for batch in train_dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Processor를 사용하여 이미지 입력 준비
        image_inputs = processor(images=images, return_tensors="pt").to(device)

        # Forward Pass (이미지 임베딩만 추출)
        outputs = model.get_image_features(**image_inputs)
        image_features = F.normalize(outputs, dim=-1)  # 이미지 임베딩 정규화

        # Similarity matrix 계산
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T  # (batch_size, num_classes)

        # Cross-Entropy Loss 계산
        loss_img = F.cross_entropy(logits_per_image, labels.long())  # labels가 Long 타입이어야 함

        # Backward & Optimization
        optimizer.zero_grad()
        loss_img.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss_img.item():.4f}")

# 7. 학습된 모델 저장
model.save_pretrained("mobileclip_lora")
processor.save_pretrained("mobileclip_lora")

# 8. 학습된 모델 로드 및 추론 테스트
model = CLIPModel.from_pretrained("mobileclip_lora")
processor = CLIPProcessor.from_pretrained("mobileclip_lora")

test_image = torch.randn(3, 224, 224)  # 테스트용 더미 이미지
image_inputs = processor(images=[test_image], return_tensors="pt").to(device)

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=-1)
    similarity_scores = image_features @ text_features.T  # 유사도 계산

print("Similarity scores:", similarity_scores)
```

---

## **🛠 주요 개선 사항**
### ✅ **1. MobileCLIP의 Text Encoder를 미리 계산**
- `model.get_text_features(**text_inputs)`를 사용하여 텍스트 임베딩을 **한 번만 계산 후 저장**합니다.
- 학습 루프에서는 **미리 계산된 `text_features`를 재사용**하여 효율성을 높였습니다.

### ✅ **2. Contrastive Loss를 수동 계산**
- `logits_per_image = logit_scale * image_features @ text_features.T` 방식으로 `logits_per_image`를 계산
- `F.cross_entropy(logits_per_image, labels.long())`를 사용하여 cross-entropy loss 계산

### ✅ **3. `F.normalize()` 적용하여 임베딩 정규화**
- `image_features`와 `text_features`를 `F.normalize(..., dim=-1)`로 정규화하여 CLIP 스타일의 similarity 계산을 유지

### ✅ **4. `labels`를 `LongTensor`로 변환**
- `F.cross_entropy()`에서 `labels`는 `LongTensor`여야 하므로 `labels.long()`을 적용하여 오류 방지

---

## **🚀 기대 효과**
✅ **Text Encoder는 미리 계산하여 재사용하므로 학습 속도 향상**  
✅ **Contrastive Loss 수동 계산을 적용하여 `loss.backward()` 오류 없이 학습 가능**  
✅ **MobileCLIP에 LoRA 적용하여 가볍게 훈련 가능**  
✅ **학습 후 모델을 저장하고 다시 로드하여 추론 가능**

이제 이 코드로 MobileCLIP을 LoRA와 함께 학습할 수 있어요! 🚀😊
