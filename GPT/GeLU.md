## 요약
**GeLU(Gaussian Error Linear Unit)** 함수는 ReLU와 같은 활성화 함수 중 하나로, 신경망에서 비선형성을 추가하는 데 사용됩니다. GeLU는 입력을 가우시안 분포를 활용하여 부드럽게 활성화하며, 특히 Transformer 모델(BERT 등)에서 많이 사용됩니다. 수식은 다음과 같습니다.

\[
\text{GeLU}(x) = x \cdot \Phi(x)
\]

여기서 \(\Phi(x)\)는 가우시안 누적 분포 함수(CDF)입니다. 이는 근사적으로 다음과 같이 표현됩니다.

\[
\text{GeLU}(x) \approx \frac{x}{2} \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)
\]

ReLU보다 매끄럽고, Swish 함수와 유사하지만 더 강력한 성능을 보이는 경우가 많습니다.

---

## 1. GeLU란?
GeLU(Gaussian Error Linear Unit)는 딥러닝에서 사용되는 **비선형 활성화 함수** 중 하나입니다. 기존의 ReLU나 Sigmoid, Tanh 같은 활성화 함수와 달리, GeLU는 **확률적 요소를 포함하여 더 부드러운 형태**를 가집니다.  
2016년 논문 ["Gaussian Error Linear Units (GELUs)"](https://arxiv.org/abs/1606.08415)에서 처음 제안되었으며, 이후 **BERT(Bidirectional Encoder Representations from Transformers)** 같은 최신 Transformer 모델에서 널리 사용되었습니다.

---

## 2. GeLU의 수식
GeLU는 입력 \(x\)를 특정 확률로 활성화하는 방식으로 작동합니다. 수학적으로는 다음과 같이 정의됩니다.

\[
\text{GeLU}(x) = x \cdot \Phi(x)
\]

여기서 \(\Phi(x)\)는 가우시안 누적 분포 함수(CDF, Cumulative Distribution Function)이며, 다음과 같이 정의됩니다.

\[
\Phi(x) = \frac{1}{2} \left( 1 + \text{erf} \left( \frac{x}{\sqrt{2}} \right) \right)
\]

여기서 **erf(x)**는 **오차 함수(Error Function)** 입니다.

이 식을 보면, **입력 \(x\)가 클수록 출력을 거의 그대로 전달하고, \(x\)가 작을수록 점진적으로 0으로 수렴**하는 것을 알 수 있습니다. 즉, ReLU처럼 **단순히 0 또는 x**로 나누지 않고, **확률적으로 활성화하는 방식**입니다.

---

## 3. 근사 표현
위의 수식은 연산량이 많기 때문에, 보다 효율적인 근사 표현이 자주 사용됩니다. 대표적인 근사식은 다음과 같습니다.

\[
\text{GeLU}(x) \approx \frac{x}{2} \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)
\]

이 근사는 실제 GeLU 함수와 거의 동일한 동작을 하면서도 **연산량을 줄일 수 있어** 실전에서 많이 사용됩니다.

---

## 4. GeLU vs. 다른 활성화 함수 비교
GeLU는 여러 활성화 함수와 비교할 때 다음과 같은 특징을 가집니다.

| 함수  | 수식  | 주요 특징 |
|------|------|--------|
| **ReLU**  | \( \max(0, x) \)  | 간단하고 계산량 적음, 하지만 미분 불연속점이 있음 (Dying ReLU 문제) |
| **Leaky ReLU** | \( \max(0.01x, x) \) | ReLU의 단점을 보완하여 음수값도 활성화 |
| **Sigmoid** | \( \frac{1}{1 + e^{-x}} \) | 출력 범위 (0,1), 하지만 기울기 소실 문제 있음 |
| **Tanh** | \( \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | 출력 범위 (-1,1), 중심이 0이라 Sigmoid보다 좋음 |
| **Swish** | \( x \cdot \sigma(x) \) | GeLU와 유사하며, ReLU보다 부드러움 |
| **GeLU** | \( x \cdot \Phi(x) \) | 확률적 활성화, Swish와 유사하지만 더 좋은 성능을 보임 |

특히, **ReLU와 비교했을 때** GeLU는 더 부드러운 활성화가 가능하여 **훈련 과정에서 더 나은 수렴성**을 보이며, Transformer 모델 등에서 널리 사용됩니다.

---

## 5. GeLU의 장점과 단점
### ✅ 장점
1. **부드러운 활성화 함수**  
   - ReLU는 0에서 불연속적인 변화가 있지만, GeLU는 부드럽게 변하기 때문에 최적화가 더 잘 이루어집니다.
   
2. **ReLU보다 강력한 성능**  
   - 여러 실험에서 ReLU보다 GeLU가 더 좋은 성능을 보였으며, 특히 **Transformer 기반 모델(BERT 등)** 에서 표준으로 사용됩니다.
   
3. **Swish보다 계산 효율적**  
   - Swish는 \( x \cdot \sigma(x) \)를 사용하므로 추가적인 exp 연산이 필요하지만, GeLU는 근사식을 이용하면 효율적입니다.

### ❌ 단점
1. **계산량 증가**  
   - ReLU(\( \max(0, x) \))보다 복잡한 연산을 수행하기 때문에 속도가 약간 느릴 수 있음.
   
2. **하드웨어 최적화 부족**  
   - ReLU는 하드웨어 최적화가 잘 되어 있지만, GeLU는 상대적으로 덜 최적화되어 있을 수 있음.

---

## 6. PyTorch에서 GeLU 사용법
PyTorch에서는 간단하게 `torch.nn.GELU`를 사용하여 GeLU를 활성화할 수 있습니다.

```python
import torch
import torch.nn as nn

gelu = nn.GELU()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = gelu(x)

print(output)
```

또는, 직접 근사식을 사용할 수도 있습니다.

```python
def gelu_approx(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / 3.141592)) * (x + 0.044715 * x**3)))

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = gelu_approx(x)

print(output)
```

---

## 7. GeLU가 주로 사용되는 곳
1. **Transformer 모델 (BERT, GPT 등)**  
   - BERT 논문에서는 ReLU 대신 GeLU를 사용하여 성능을 향상시켰음.
   
2. **Vision Transformer (ViT)**  
   - 이미지 처리에서 Transformer 구조를 사용할 때도 GeLU가 적용됨.
   
3. **일반적인 신경망 최적화**  
   - CNN, RNN 등 다양한 네트워크에서 ReLU보다 성능이 좋은 경우 활용.

---

## 8. 결론
GeLU는 활성화 함수 중 하나로, 확률적 활성화를 통해 더 부드러운 비선형성을 제공합니다. 특히, Transformer 모델에서 표준적으로 사용될 만큼 성능이 뛰어나며, Swish와 비슷하지만 더 나은 성능을 보이는 경우가 많습니다. 계산량이 증가하는 단점이 있지만, 딥러닝 모델의 **성능 향상을 위해 고려해볼 만한 활성화 함수**입니다.
