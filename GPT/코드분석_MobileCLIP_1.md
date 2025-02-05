```clip.py```

## 코드 분석 및 설명

이 코드는 **Apple Inc.에서 개발한 CLIP 모델의 OpenCLIP 형식 변형**으로, 이미지와 텍스트를 멀티모달 방식으로 처리하는 **추론(inference) 전용 모델**입니다. PyTorch 기반으로 작성되었으며, 이미지와 텍스트 임베딩을 생성하고, 이를 정규화하여 유사도를 계산할 수 있도록 설계되었습니다.

---

## **1. 주요 개념 및 목적**
이 코드는 OpenAI의 CLIP(Contrastive Language-Image Pretraining) 모델을 기반으로 하지만, Apple에서 만든 `MCi`라는 자체적인 이미지 인코더와 `TextTransformer`라는 텍스트 인코더를 사용하여 학습 및 추론이 가능하도록 설계되었습니다.

### **주요 기능**
- 이미지 인코딩 (Image Encoding)
- 텍스트 인코딩 (Text Encoding)
- 이미지-텍스트 임베딩 비교
- 정규화된 임베딩 벡터 생성
- `logit_scale` 조정 (유사도 계산을 위한 스케일링)

---

## **2. 코드 구조 분석**

### **(1) 클래스 및 모듈 가져오기**
```python
import math
from typing import Any, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn

from mobileclip.text_encoder import (
    TextTransformer,
)

from .image_encoder import MCi
```
- `math`: 로그 연산에 사용
- `typing`: 함수의 타입 힌트 제공
- `torch`, `torch.nn`, `torch.nn.functional`: PyTorch 기반 신경망 모델 구성
- `TextTransformer`: 텍스트 인코딩을 담당하는 Transformer 모델 (`mobileclip.text_encoder` 모듈에서 가져옴)
- `MCi`: 이미지 인코딩을 담당하는 모델 (`image_encoder.py`에서 가져옴)

---

### **(2) CLIP 클래스 정의**
```python
class CLIP(nn.Module):
    """Base class for multi-modal image-text data"""
```
- `nn.Module`을 상속받아 PyTorch의 신경망 모델 형태로 정의됨
- 이미지와 텍스트를 처리하는 **멀티모달 모델** 역할을 함

#### **(a) 생성자 (`__init__`)**
```python
def __init__(self, cfg: Dict, output_dict: bool = False, *args, **kwargs) -> None:
    super().__init__()
    self.output_dict = output_dict
    self.projection_dim = cfg["embed_dim"]
    if self.projection_dim is None:
        raise ValueError("Please specify `embed_dim` in model config.")

    self.image_encoder = MCi(
        model_name=cfg["image_cfg"]["model_name"],
        projection_dim=self.projection_dim,
    )
    self.text_encoder = TextTransformer(
        cfg=cfg["text_cfg"], projection_dim=self.projection_dim
    )
    self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
```
- `cfg` (딕셔너리): 모델 설정값을 포함 (예: `embed_dim`, `image_cfg`, `text_cfg`)
- `self.projection_dim`: 이미지 및 텍스트 임베딩 차원
- `self.image_encoder`: `MCi` 클래스 기반 이미지 인코더
- `self.text_encoder`: `TextTransformer` 기반 텍스트 인코더
- `self.logit_scale`: 초기값 `math.log(1.0 / 0.07)`을 사용해 **학습 가능한 로그 스케일 변수**를 정의  
  → 임베딩 벡터 간 **유사도를 계산할 때 사용**

---

#### **(b) logit scale 처리 함수**
```python
def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
    scale = self.logit_scale.exp()
    scale = torch.clamp(scale, 0, max_scale)
    return scale
```
- `self.logit_scale.exp()`: 로그 값(`logit_scale`)을 **지수 연산**을 통해 변환
- `torch.clamp(scale, 0, max_scale)`: 스케일 값을 최대 `100.0`으로 제한  
  → 너무 높은 값으로 인해 학습이 불안정해지는 것을 방지

---

#### **(c) 이미지 인코딩**
```python
def encode_image(self, image: torch.Tensor, normalize: bool = False):
    image_encoder_out = self.image_encoder(image)
    if isinstance(image_encoder_out, dict):
        features = image_encoder_out["logits"]
    else:
        features = image_encoder_out
    return F.normalize(features, dim=-1) if normalize else features
```
- `self.image_encoder(image)`: `MCi` 모델을 사용하여 **이미지 임베딩 벡터를 생성**
- 출력값이 `dict`일 경우 `"logits"` 값을 사용
- `normalize=True`이면 L2 정규화된 벡터 반환

---

#### **(d) 텍스트 인코딩**
```python
def encode_text(self, text: torch.Tensor, normalize: bool = False):
    text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
    return F.normalize(text_features, dim=-1) if normalize else text_features
```
- `self.text_encoder(text_tokens=text, key_padding_mask=None)`: Transformer 기반으로 텍스트 인코딩 수행
- `normalize=True`이면 정규화된 벡터 반환

---

#### **(e) 모델의 `forward()` 연산**
```python
def forward(
    self,
    image: Optional[torch.Tensor] = None,
    text: Optional[torch.Tensor] = None,
    *args,
    **kwargs
) -> Any:
```
- 이미지 또는 텍스트 중 하나만 입력할 수도 있고, 둘 다 입력 가능

```python
image_embeddings = (
    self.encode_image(image, normalize=True) if image is not None else None
)
text_embeddings = (
    self.encode_text(text, normalize=True) if text is not None else None
)
```
- 이미지와 텍스트를 각각 인코딩하여 벡터를 생성

```python
if self.output_dict:
    return {
        "image_features": image_embeddings,
        "text_features": text_embeddings,
        "logit_scale": self._exponentiate_and_clip_logits(),
    }
return image_embeddings, text_embeddings, self._exponentiate_and_clip_logits()
```
- `self.output_dict=True`이면 **딕셔너리 형태의 출력값 반환**
- 그렇지 않으면 **튜플 형태의 임베딩과 logit scale 반환**

---

## **3. 코드의 핵심 기능 요약**
1. **이미지 인코딩 (`encode_image`)**
   - `MCi` 모델을 사용하여 이미지를 **벡터로 변환**
   - 필요하면 정규화하여 벡터를 반환
2. **텍스트 인코딩 (`encode_text`)**
   - `TextTransformer`를 사용하여 텍스트를 **벡터로 변환**
   - 필요하면 정규화하여 벡터를 반환
3. **logit scale 조정 (`_exponentiate_and_clip_logits`)**
   - 지수 변환 후, 최대 100.0까지 제한하여 학습 안정성을 확보
4. **추론 (`forward`)**
   - 이미지와 텍스트 입력을 각각 처리하여 **멀티모달 임베딩 벡터** 생성
   - `output_dict=True`이면 딕셔너리로 반환, 아니면 튜플로 반환

---

## **4. 결론**
이 코드는 Apple의 `mobileclip` 기반 CLIP 모델 구현체로, **이미지와 텍스트 간 유사도를 계산**하는 것이 목적입니다.  
- PyTorch의 `nn.Module`을 활용하여 구성되었으며,  
- `MCi`(이미지 인코더)와 `TextTransformer`(텍스트 인코더)를 사용하여  
- 이미지와 텍스트를 각각 벡터로 변환하고  
- 정규화된 벡터를 비교하여 의미적 유사도를 측정할 수 있도록 합니다.

추가로 궁금한 부분이 있으면 질문 주세요! 😊
