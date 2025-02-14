아래는 MobileCLIP 모델에 LoRA를 적용하고, contrastive loss를 수동으로 계산하여 학습하는 전체 코드입니다. `peft` 라이브러리를 사용하여 LoRA를 적용하고, contrastive loss를 직접 계산하는 방식으로 구현했습니다.

---

### **MobileCLIP + LoRA + Contrastive Loss 적용 코드**
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

# 3. 데이터셋 및 데이터로더 정의 (사용자 데이터셋에 맞게 수정)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, processor, num_samples=100):
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = "A cat sitting on a chair"
        image = torch.randn(3, 224, 224)  # 가짜 이미지 데이터 (실제 이미지로 교체 필요)
        return {"text": text, "image": image}

train_dataset = DummyDataset(processor)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. 옵티마이저 및 학습 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# 5. Contrastive Loss를 수동 계산하며 모델 학습
for epoch in range(3):  # 3 Epoch 예제
    for batch in train_dataloader:
        images = batch["image"].to(device)
        texts = batch["text"]

        # Processor를 사용하여 입력 준비
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

        # Forward Pass
        outputs = model(**inputs)
        image_features = outputs.image_embeds  # 이미지 임베딩
        text_features = outputs.text_embeds  # 텍스트 임베딩

        # Normalize embeddings
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Similarity matrix
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Create labels
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size).to(device)

        # Contrastive Loss (CLIP Loss)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        # Backward & Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 6. 학습된 모델 저장
model.save_pretrained("mobileclip_lora")
processor.save_pretrained("mobileclip_lora")

# 7. 학습된 모델 로드 및 추론 테스트
model = CLIPModel.from_pretrained("mobileclip_lora")
processor = CLIPProcessor.from_pretrained("mobileclip_lora")

text_inputs = ["a cat", "a dog"]
image_inputs = torch.randn(3, 224, 224)  # 테스트용 더미 이미지
inputs = processor(text=text_inputs, images=[image_inputs], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도 점수

print(logits_per_image)
```

---

### **설명**
1. **MobileCLIP 모델 로드**  
   - Apple에서 제공하는 `ml-mobileclip` 모델을 로드합니다. (`from_pretrained` 사용)

2. **LoRA 적용**  
   - `peft` 라이브러리의 `LoraConfig`를 이용하여 `text_model.encoder.layers`와 `vision_model.encoder.layers`에 LoRA를 적용합니다.

3. **Contrastive Loss 수동 계산**  
   - 모델의 `image_embeds`와 `text_embeds`를 가져와서 정규화합니다.
   - Cosine similarity를 계산한 후, `cross_entropy`를 이용하여 Contrastive Loss를 직접 계산합니다.

4. **모델 학습 루프**  
   - `AdamW` 옵티마이저를 사용하여 3 Epoch 동안 학습합니다.

5. **학습된 모델 저장 및 추론**  
   - 학습이 완료된 모델을 저장하고, 이후 다시 불러와서 테스트할 수 있도록 구성했습니다.

---

### **이 코드의 특징**
✅ **LoRA 적용**: MobileCLIP 모델을 가볍게 학습 가능  
✅ **Contrastive Loss 수동 계산**: `outputs.loss` 없이도 CLIP 방식의 contrastive loss 직접 계산  
✅ **학습 후 모델 저장 및 추론**: LoRA 적용 후 저장한 모델을 불러와서 테스트 가능  

이제 이 코드를 실행하면 MobileCLIP에 LoRA를 적용하고, Contrastive Loss를 활용하여 학습할 수 있습니다! 🚀

---
---
좋은 질문이에요! `loss.backward()`가 어떻게 동작하는지 이해하려면 **PyTorch의 자동 미분(Autograd) 메커니즘**을 알아야 해요. 아래에서 이를 단계적으로 설명할게요.

---

## **1. PyTorch의 Autograd (자동 미분)**
PyTorch에서 `Tensor`는 `requires_grad=True` 속성을 가지고 있을 경우, 모든 연산의 그래프를 자동으로 생성해요. 이 그래프를 통해 **역전파(Backpropagation)**가 가능해지고, `loss.backward()`가 호출되면, 이 그래프를 따라 미분이 전파돼요.

---

## **2. `loss.backward()`의 동작 과정**
1. **연산 그래프 구성**
   - `logits_per_image = logit_scale * image_features @ text_features.T`
   - `loss_img = F.cross_entropy(logits_per_image, labels)`
   - `loss_txt = F.cross_entropy(logits_per_text, labels)`
   - `loss = (loss_img + loss_txt) / 2`

   이 연산들은 모두 `requires_grad=True`인 텐서(`model`의 가중치들 포함)를 포함하고 있기 때문에 PyTorch는 자동으로 **계산 그래프**를 만들어요.

2. **역전파 (Backward Pass)**
   ```python
   loss.backward()
   ```
   이 코드를 실행하면:
   - `loss`에서 출발하여 **연산 그래프를 따라 역전파**가 수행돼요.
   - `loss`는 `loss_img`와 `loss_txt`의 평균으로 구성되었기 때문에, 먼저 이 두 개의 gradient가 계산돼요.
   - `loss_img`와 `loss_txt`는 각각 `logits_per_image`와 `logits_per_text`에 대한 cross-entropy loss이므로, 이 값들에 대한 gradient가 계산돼요.
   - 이 과정이 계속 거슬러 올라가면서 `image_features`와 `text_features`, 그리고 모델 내부의 `encoder.layers`까지 gradient가 전달돼요.

3. **모든 `requires_grad=True`인 텐서들이 Gradient 업데이트 가능**
   - `loss.backward()`를 호출하면, `model.parameters()`에 있는 모든 **학습 가능한 파라미터(가중치들)**가 `grad` 값을 갖게 돼요.
   - 이후 `optimizer.step()`을 실행하면, 옵티마이저가 이 `grad` 값을 기반으로 가중치를 업데이트해요.

---

## **3. 지역 변수인데 왜 전달되나요?**
`loss`는 `loss.backward()`를 호출한 순간 **더 이상 필요하지 않아요!**
- `loss`는 연산 그래프의 가장 마지막 부분이고, `loss.backward()`를 호출하면 PyTorch는 **연산 그래프를 따라 역전파를 진행**해요.
- 따라서 `loss`가 지역 변수이더라도, **모델 내부 가중치들(`model.parameters()`)에 연결된 연산 그래프가 남아 있어서**, 해당 가중치들의 `grad`가 계산될 수 있어요.

즉, **`loss` 자체가 전달되는 게 아니라**, **`loss`가 계산된 과정에서 생긴 연산 그래프를 통해 gradient가 역전파되는 거예요!** 🚀

---

## **4. 정리**
✅ `loss.backward()`는 **계산 그래프를 따라 역전파를 수행**한다.  
✅ `loss`는 단순한 변수이지만, **그 연산에 사용된 모든 텐서는 PyTorch의 Autograd 시스템을 통해 연결**되어 있다.  
✅ 따라서, `loss`가 지역 변수라도 **역전파는 모델의 가중치까지 전달**될 수 있다.

이제 `loss.backward()`가 어떻게 gradient를 업데이트하는지 더 명확해졌죠? 😊
