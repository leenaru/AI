아래는 **프롬프트 튜닝(Prompt Tuning)**에 대한 개념과, 이를 **LoRA** 및 **Full Fine-tuning** 등 다른 미세조정 방법과 비교한 내용을 요약한 후 상세하게 설명한 내용입니다.

---

### 요약

- **프롬프트 튜닝**은 입력에 추가되는 임베딩(프롬프트)을 학습하여 모델의 출력을 조정하는 기법으로, 모델 내부의 파라미터는 고정한 채 매우 적은 수의 파라미터만 업데이트합니다.  
- **LoRA**는 모델 내부의 특정 레이어(예, 어텐션 계층)의 가중치에 저차원 업데이트(low-rank matrix)를 추가하는 방식으로, 모델의 핵심 연산 흐름을 보존하면서도 효과적으로 미세조정할 수 있습니다.  
- **Full Fine-tuning**은 모델의 전체 파라미터를 업데이트하는 방식으로, 가장 유연하지만 자원 소모와 과적합 위험이 큰 반면, 프롬프트 튜닝과 LoRA는 **파라미터 효율적 미세조정(PEFT)** 방법으로 자원 부담을 줄여줍니다.  
- **프롬프트 튜닝의 장점**은 매우 가볍고, 학습해야 할 파라미터 수가 극히 적어 자원 소모가 작다는 점이며, **단점**은 모델의 깊은 내부 표현을 수정하지 못해 일부 복잡한 태스크에 한계가 있을 수 있다는 점입니다.

---

### 1. 프롬프트 튜닝(Prompt Tuning)의 개념

- **정의:**  
  프롬프트 튜닝은 사전학습된 모델에 대해 입력 프롬프트(일종의 가변 임베딩 벡터)를 추가하고, 이 프롬프트를 학습하여 모델이 주어진 다운스트림 태스크에 맞게 동작하도록 유도하는 방법입니다.  
  예를 들어, 텍스트 생성 모델에서는 입력 문장의 앞부분에 학습된 프롬프트 벡터를 덧붙여 모델이 특정한 스타일이나 태스크에 맞춰 응답하도록 할 수 있습니다.

- **특징:**  
  - 모델 내부의 파라미터는 변경하지 않고, 오직 프롬프트 임베딩만 학습합니다.  
  - 매우 소수의 파라미터만 업데이트하기 때문에 저장 공간 및 연산 자원 측면에서 효율적입니다.  
  - 주로 대형 언어모델이나 멀티모달 모델의 경우, 사전학습된 지식을 그대로 유지하면서 태스크에 특화된 정보를 보완하는 방식으로 사용됩니다.

---

### 2. 다른 미세조정 방법과의 비교

#### (1) Full Fine-tuning  
- **설명:**  
  - 모델의 모든 파라미터를 업데이트하는 방식입니다.
- **장점:**  
  - 태스크 특성에 맞춰 모델의 전체 표현을 미세하게 조정할 수 있어, 충분한 데이터와 자원이 있을 경우 최고 성능을 낼 가능성이 높습니다.
- **단점:**  
  - 전체 파라미터를 업데이트하기 때문에 GPU VRAM, 연산 자원, 저장 공간 등이 많이 필요하며, 개인 PC에서는 부담이 될 수 있습니다.
  - 파라미터 수가 많아 과적합(overfitting) 위험이 존재할 수 있습니다.

#### (2) LoRA (Low-Rank Adaptation)  
- **설명:**  
  - 모델 내부의 특정 레이어(예, 어텐션 모듈)에 대해 저차원 행렬을 추가해 업데이트함으로써, 전체 파라미터 중 일부만 학습하는 기법입니다.
- **장점:**  
  - 프롬프트 튜닝보다 모델의 내부 계산 흐름에 직접 개입하므로, 더 정교한 미세조정이 가능하며 성능 향상에 유리할 수 있습니다.
  - 전체 모델을 업데이트하지 않아 자원 절감 효과가 있으며, 특히 큰 모델에 적용할 때 효과적입니다.
- **단점:**  
  - 프롬프트 튜닝보다는 추가적인 연산 및 메모리 오버헤드가 발생할 수 있으며, 적용할 레이어나 하이퍼파라미터 설정에 따라 성능 변동이 존재합니다.

#### (3) 프롬프트 튜닝 (Prompt Tuning)  
- **설명:**  
  - 입력 부분에 추가되는 프롬프트 벡터만 학습하는 방식입니다.
- **장점:**  
  - 업데이트해야 하는 파라미터 수가 극히 적어, 매우 가볍고 빠른 학습이 가능합니다.
  - 저장 및 전송 시에도 최소의 추가 용량만 필요하며, 자원 제한 환경에서 효율적입니다.
- **단점:**  
  - 모델 내부의 표현을 직접 수정하지 않기 때문에, 태스크에 필요한 세밀한 조정이 어려울 수 있습니다.
  - 프롬프트의 효과가 태스크와 모델의 사전학습 특성에 크게 의존하므로, 모든 상황에서 최적의 성능을 보장하지는 않습니다.

---

### 3. 비교 정리 및 선택 기준

- **자원 사용 측면:**  
  - **프롬프트 튜닝**은 학습해야 하는 파라미터 수가 매우 적어, 메모리와 저장 공간 면에서 가장 효율적입니다.  
  - **LoRA**도 Full Fine-tuning에 비해 자원 소모가 적지만, 프롬프트 튜닝보다는 약간의 추가 오버헤드(저차원 업데이트 행렬 등)가 있습니다.
  - **Full Fine-tuning**은 자원 요구량이 가장 높습니다.

- **성능 및 유연성 측면:**  
  - **Full Fine-tuning**은 충분한 데이터와 자원이 있을 경우, 모든 파라미터를 조정하므로 최고 성능을 낼 수 있습니다.
  - **LoRA**는 모델 내부에서 중요한 부분만 수정하기 때문에, 태스크에 따라 프롬프트 튜닝보다 더 나은 성능을 보일 수 있습니다.
  - **프롬프트 튜닝**은 모델의 사전학습된 지식을 최대한 활용하면서 가벼운 미세조정을 할 때 유리하지만, 태스크에 따라서는 모델 내부의 변화가 필요한 경우 한계가 있을 수 있습니다.

- **적용 분야:**  
  - **프롬프트 튜닝**은 특히 대형 언어 모델에서, 입력 문맥이나 스타일 조정이 중요한 태스크(예: 텍스트 생성, 번역 등)에 효과적입니다.
  - **LoRA**는 언어, 이미지, 멀티모달 등 다양한 도메인에서 미세조정이 필요할 때 유연하게 적용할 수 있습니다.
  - **Full Fine-tuning**은 자원과 데이터가 충분한 경우에만 고려하는 방법으로, 자원 효율성 측면에서는 덜 선호됩니다.

---

### 4. 결론

개인 PC와 같이 자원이 제한된 환경에서는 **프롬프트 튜닝**과 **LoRA** 모두 훌륭한 파라미터 효율적 미세조정 기법입니다.  
- **프롬프트 튜닝**은 업데이트해야 하는 파라미터 수가 매우 적어 빠르고 가볍게 학습할 수 있지만, 태스크에 따라 모델 내부의 복잡한 표현을 변경하는 데는 한계가 있을 수 있습니다.  
- **LoRA**는 모델 내부의 중요한 부분만 수정하여 더 정교한 미세조정을 가능하게 하며, 프롬프트 튜닝보다 더 나은 성능을 발휘할 가능성이 있지만, 약간의 추가 자원 사용이 필요합니다.

따라서, 작업하는 태스크의 특성과 모델의 사전학습 상태, 그리고 자원 상황에 따라 선택하면 되겠습니다.  
- 만약 자원이 극도로 제한되어 있고, 태스크가 프롬프트로 충분히 유도 가능한 경우라면 **프롬프트 튜닝**이 좋은 선택입니다.  
- 보다 복잡하거나 세밀한 태스크의 경우에는 **LoRA**를 적용하여 모델 내부의 표현을 조정하는 것이 효과적일 수 있습니다.

이와 같이 각 방법의 장단점을 고려해 상황에 맞는 미세조정 전략을 선택하는 것이 중요합니다.


# -----------------------------
아래는 MobileCLIP(혹은 경량 CLIP 모델)에 **프롬프트 튜닝(Prompt Tuning)**을 적용하는 예시 코드와 함께, 코드가 작동하는 원리 및 각 부분의 설명을 자세히 정리한 내용입니다.

---

### 요약

- **프롬프트 튜닝**은 모델 내부의 파라미터는 고정한 채, 입력 임베딩 앞에 추가되는 학습 가능한 프롬프트(또는 prefix) 토큰만을 업데이트하는 방법입니다.  
- MobileCLIP과 같이 CLIP 계열의 경량 모델에도, 텍스트 인코더의 입력에 프롬프트 임베딩을 추가하여 파인튜닝할 수 있습니다.  
- 아래 예시에서는 Hugging Face의 CLIP 모델(예: `openai/clip-vit-base-patch32`)을 MobileCLIP의 대체 모델로 가정하고, 텍스트 인코더에 프롬프트 튜닝을 적용하는 방법을 설명합니다.

---

### 코드 예시 및 상세 설명

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class PromptTuningCLIP(nn.Module):
    """
    프롬프트 튜닝을 적용한 CLIP 모델 래퍼.
    - 입력 텍스트의 임베딩 앞에 learnable prompt 토큰들을 추가합니다.
    - MobileCLIP(또는 경량 CLIP) 모델의 나머지 파라미터는 고정(freeze)합니다.
    """
    def __init__(self, model, prompt_length=5):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length

        # 토크나이저 로드 (모델에 맞게 사용)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 모델 전체 파라미터를 동결시킵니다.
        for param in self.model.parameters():
            param.requires_grad = False

        # 텍스트 인코더의 임베딩 차원에 맞는 learnable prompt 임베딩을 초기화합니다.
        # hidden_size는 모델의 텍스트 구성 설정에서 확인합니다.
        hidden_size = self.model.config.text_config.hidden_size
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, hidden_size))

    def forward(self, input_text, images):
        """
        input_text: 리스트 형태의 문자열 (예: ["a photo of a cat", "a photo of a dog"])
        images: 텐서 형태의 이미지 배치 (예: [batch_size, 3, 224, 224])
        """
        device = self.prompt_embeddings.device

        # 텍스트 토큰화
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        # 텍스트 인코더의 기본 임베딩 추출
        # CLIP 모델은 텍스트 인코더에 "inputs_embeds"를 전달할 수 있습니다.
        # embeddings: [batch_size, seq_len, hidden_size]
        text_embeddings = self.model.text_model.embeddings(input_ids)

        batch_size = text_embeddings.shape[0]

        # 학습 가능한 프롬프트 임베딩을 배치 차원에 맞게 확장
        # prompt_embed: [batch_size, prompt_length, hidden_size]
        prompt_embed = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # 원래 임베딩 앞에 프롬프트 임베딩을 concatenate 합니다.
        # combined_embeddings: [batch_size, prompt_length + seq_len, hidden_size]
        combined_embeddings = torch.cat([prompt_embed, text_embeddings], dim=1)

        # attention mask도 프롬프트 토큰에 대해 1을 추가하여 확장
        prompt_mask = torch.ones(batch_size, self.prompt_length, device=device)
        combined_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 텍스트 인코더에 입력 시, "inputs_embeds"를 사용하여 미리 조합한 임베딩을 전달합니다.
        # (모델 구조에 따라 forward() 호출 방식은 다를 수 있으므로, 실제 MobileCLIP에서는 해당 부분을 확인해야 합니다.)
        text_outputs = self.model.text_model.encoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
        )

        # 일반적으로 텍스트 인코더의 출력 중 [CLS] 토큰 혹은 pooled output을 사용합니다.
        # 여기서는 첫 토큰(프롬프트를 포함하므로)의 평균값을 예시로 사용합니다.
        text_features = text_outputs.last_hidden_state.mean(dim=1)

        # 이미지 인코더 처리 (일반적인 처리 방식)
        image_outputs = self.model.vision_model(images)
        image_features = image_outputs[1]  # 보통 pooled output 사용

        # 텍스트와 이미지 임베딩 간의 유사도(예: 내적)를 계산하여 최종 로짓 산출
        logits = (text_features @ image_features.T)

        return logits

# 사용 예시
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 예시로 CLIP 모델 로드 (실제 MobileCLIP이 공개되어 있다면 해당 모델을 사용)
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    base_model.to(device)

    # 프롬프트 튜닝을 적용한 모델 초기화 (prompt_length는 원하는 프롬프트 토큰 수)
    prompt_model = PromptTuningCLIP(base_model, prompt_length=5).to(device)

    # 더미 데이터: 텍스트와 이미지 배치
    dummy_text = ["a photo of a cat", "a photo of a dog"]
    dummy_images = torch.randn(2, 3, 224, 224).to(device)

    # 모델 순전파 (forward pass)
    logits = prompt_model(dummy_text, dummy_images)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    # 이후 loss 계산 및 optimizer를 이용한 파인튜닝 진행
    # 단, 모델의 프롬프트 임베딩 파라미터만 업데이트됩니다.
```

---

### 코드 설명

1. **모델 및 토크나이저 로드**  
   - Hugging Face의 CLIP 모델과 해당 토크나이저를 불러옵니다.  
   - MobileCLIP의 경우에도 동일한 구조를 가정하거나, 경량화된 CLIP 버전을 사용할 수 있습니다.

2. **모델 동결 및 프롬프트 임베딩 초기화**  
   - 모델 전체의 파라미터는 `requires_grad = False`로 설정하여 고정합니다.  
   - 텍스트 인코더의 임베딩 차원에 맞춰, 지정한 길이(`prompt_length`)의 프롬프트 임베딩을 학습 가능한 파라미터로 초기화합니다.

3. **입력 임베딩에 프롬프트 임베딩 추가**  
   - 입력 텍스트를 토큰화하고, 원래 임베딩을 구한 후, 프롬프트 임베딩을 앞에 concatenation합니다.  
   - attention mask도 프롬프트에 대해 1로 확장하여 모델이 프롬프트 부분을 “주의”하도록 합니다.

4. **텍스트 인코더와 이미지 인코더 처리**  
   - 조합된 임베딩을 텍스트 인코더에 입력하여 텍스트 특성을 얻고, 이미지 인코더는 일반 방식으로 처리합니다.
   - 텍스트와 이미지 임베딩의 내적 등을 통해 유사도를 계산합니다.

5. **파인튜닝**  
   - 이후 loss(예: contrastive loss 등)를 계산하여 프롬프트 임베딩만 업데이트하는 방식으로 학습을 진행하면 됩니다.
   - 이 방식은 전체 모델 파라미터를 업데이트하지 않으므로, 적은 자원으로도 파인튜닝이 가능합니다.

---

### 결론

이 예시는 MobileCLIP(혹은 경량 CLIP 모델)에 프롬프트 튜닝을 적용하는 한 가지 방법을 보여줍니다.  
- 전체 모델 파라미터를 고정하고, 입력 임베딩 앞에 학습 가능한 프롬프트를 추가하여 텍스트 인코더를 미세조정합니다.  
- 이 방식은 자원 효율적이며, 특히 제한된 하드웨어 환경(예: 개인 PC)에서 빠르게 파인튜닝을 진행할 때 유용합니다.

실제 MobileCLIP 모델의 구조에 따라 약간의 수정이 필요할 수 있으므로, 모델 문서를 참고하여 적절히 조정하시기 바랍니다.
