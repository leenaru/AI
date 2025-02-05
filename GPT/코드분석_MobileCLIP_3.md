# 📌 **코드 분석 및 설명 (추가 파일 포함)**

이번에 업로드된 추가 파일들은 MobileCLIP의 **Zero-shot ImageNet 평가, 데이터 변환 및 증강(Augmentation) 기법**을 담당하는 코드들입니다.  
각 파일이 어떤 역할을 하는지 상세히 분석해보겠습니다.

---

## **📂 추가된 파일 개요**
| 파일명 | 주요 역할 |
|--------|----------|
| `zeroshot_imagenet.py` | MobileCLIP을 사용한 **Zero-shot ImageNet 평가** |
| `transforms.py` | PyTorch 기반 **데이터 변환(Transformations) 모듈** |
| `transforms_base.py` | 데이터 증강(Augmentation) 관련 기본 변환 모음 |

---

# **1️⃣ `zeroshot_imagenet.py` 분석 (Zero-shot ImageNet 평가)**
📌 **역할:**  
- MobileCLIP을 활용하여 **ImageNet에서 Zero-shot 학습된 모델을 평가**
- `clip_benchmark` 라이브러리 사용

### 🔹 **핵심 코드 분석**
#### **(1) 인자 파싱**
```python
def parse_args(parser):
    parser.add_argument(
        "--model-arch",
        type=str,
        required=True,
        choices=['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    return parser
```
- 실행 시 **사용할 MobileCLIP 모델(`--model-arch`)과 모델 체크포인트(`--model-path`)를 입력받음**  
- `mobileclip_s0`, `mobileclip_s1` 등 다양한 모델 구조를 지원

---

#### **(2) MobileCLIP 모델 불러오기**
```python
def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = mobileclip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval().to(device)
    return model, transform, device
```
- MobileCLIP 모델을 로드한 후 **추론 모드로 전환(`eval()`)**
- **이미지 전처리(`transform`)도 함께 반환**

---

#### **(3) ImageNet 데이터셋 로드**
```python
def create_webdataset(task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    dataset = build_dataset(dataset_name=f"wds/{task}", root=data_root, transform=transform, split="test")
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size), batch_size=None, shuffle=False, num_workers=num_workers
    )
    return dataset, dataloader
```
- `Hugging Face`에서 ImageNet 데이터를 가져와 평가
- `batch_size=64`로 데이터를 배치 단위로 로딩

---

#### **(4) 모델 평가 수행**
```python
def evaluate_webdataset(task, model_arch, model_path, data_root=None, batch_size=64):
    model, transform, device = create_model(model_arch, model_path)
    dataset, dataloader = create_webdataset(task, transform, data_root, batch_size)

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None

    classifier = zsc.zero_shot_classifier(model, mobileclip.get_tokenizer(model_arch), classnames, zeroshot_templates, device)
    logits, target = zsc.run_classification(model, classifier, dataloader, device, amp=False)

    acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    return {"acc1": acc1, "acc5": acc5}
```
- `zero_shot_classifier()`를 사용하여 MobileCLIP을 활용한 **Zero-shot 학습 평가** 진행  
- `top-1`과 `top-5` 정확도(`acc1`, `acc5` 메트릭) 계산  

✅ **정리:**  
- 이 파일은 **MobileCLIP 모델의 Zero-shot 성능을 ImageNet에서 평가**하는 역할을 함  
- 평가 데이터는 `Hugging Face`에서 불러오며, 모델 성능을 측정하기 위해 `top-1`과 `top-5` 정확도를 계산  

---

# **2️⃣ `transforms.py` 분석 (데이터 변환 모듈)**
📌 **역할:**  
- `torchvision.transforms`의 기본 기능을 확장하여 **이미지 변환 및 증강(Augmentation)을 보다 세밀하게 제어**
- `Compressible` 클래스를 도입하여 **변환된 이미지를 다시 변환 가능하도록 설계**

### 🔹 **핵심 코드 분석**
#### **(1) `Compressible` 클래스**
```python
class Compressible:
    @staticmethod
    def compress_params(params: Any) -> Any:
        return params

    @staticmethod
    def decompress_params(params: Any) -> Any:
        return params
```
- 이미지 변환의 랜덤한 요소(예: `RandomCrop`)를 **다시 적용할 수 있도록 매개변수를 저장 및 압축하는 기능** 제공

---

#### **(2) `Resize` 및 `CenterCrop` 변환**
```python
class Resize(T.Resize, Compressible):
    def forward(self, img: Tensor, params: Optional[torch.Size] = None):
        img = super().forward(img)
        return img, self.size

class CenterCrop(T.CenterCrop, Compressible):
    def forward(self, img: Tensor, params: Optional[Tuple[int, int]] = None):
        img = super().forward(img)
        return img, NO_PARAM
```
- `torchvision.transforms.Resize` 및 `CenterCrop`을 확장하여 **변환 정보를 저장할 수 있도록 개선**  
- 이미지 크기 조정 후, 변환된 크기를 반환하여 **재적용 가능**

✅ **정리:**  
- 이 파일은 **MobileCLIP의 데이터 전처리를 보다 세밀하게 컨트롤하는 기능**을 제공  
- `Resize`, `CenterCrop`, `RandomCrop` 등 변환이 다시 적용될 수 있도록 매개변수를 저장  

---

# **3️⃣ `transforms_base.py` 분석 (기본 이미지 변환)**
📌 **역할:**  
- `AutoAugment`, `RandAugment`, `TrivialAugmentWide` 등의 **데이터 증강(Augmentation) 기법을 포함**
- MobileCLIP 모델 학습 시, **더욱 강력한 데이터 변환 기능을 지원**

### 🔹 **핵심 코드 분석**
#### **(1) AutoAugment / RandAugment**
```python
class AutoAugment(T.AutoAugment):
    def __init__(self, policy: str = "imagenet", interpolation: str = "bilinear"):
        policy = T.AutoAugmentPolicy.IMAGENET
        interpolation = INTERPOLATION_MODE_MAP[interpolation]
        super().__init__(policy=policy, interpolation=interpolation)
```
- **자동 데이터 증강(AutoAugment)** 적용 (ImageNet, CIFAR, SVHN 등 지원)

```python
class RandAugment(T.RandAugment):
    def __init__(self, interpolation: str = "bilinear"):
        interpolation = INTERPOLATION_MODE_MAP[interpolation]
        super().__init__(interpolation=interpolation)
```
- **랜덤 데이터 증강(RandAugment)** 적용

✅ **정리:**  
- 이 파일은 MobileCLIP 학습을 위한 **고급 데이터 증강(Augmentation) 기법을 포함**  
- `AutoAugment`, `RandAugment`를 활용하여 **데이터 다양성을 극대화**  

---

# **📌 최종 정리**
| 파일명 | 역할 |
|--------|------|
| `zeroshot_imagenet.py` | MobileCLIP을 사용한 **Zero-shot ImageNet 평가** |
| `transforms.py` | PyTorch의 `torchvision.transforms`를 확장하여 **이미지 변환 가능하도록 수정** |
| `transforms_base.py` | MobileCLIP의 **데이터 증강(Augmentation) 기법**을 제공 |

---

✅ **결론:**  
- MobileCLIP의 **Zero-shot 학습된 모델을 평가하는 코드**(`zeroshot_imagenet.py`)  
- MobileCLIP의 **데이터 변환 및 증강(Augmentation) 모듈** (`transforms.py`, `transforms_base.py`)  
- 학습 및 평가 과정에서 **이미지 변환을 재적용할 수 있도록 설계된 구조**

추가 질문이 있다면 언제든지 물어보세요! 😊
