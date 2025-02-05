### **요약**
Vision dataset을 만드는 과정은 크게 5단계로 나눌 수 있습니다.

1. **목적 정의 및 데이터 수집**
   - 목표를 설정하고, 사용할 데이터의 종류(이미지, 동영상)를 정함.
   - 직접 촬영, 크롤링, 오픈소스 데이터 활용 등으로 데이터 수집.

2. **데이터 전처리**
   - 이미지 정리(중복 제거, 품질 확인, 포맷 변환).
   - 레이블링(수동/자동 라벨링, Bounding Box, Segmentation).

3. **데이터 증강(Augmentation)**
   - 회전, 색 변환, 크롭 등으로 데이터 다양성 증가.

4. **데이터셋 구조화 및 저장**
   - 학습/검증/테스트 데이터 분할.
   - 적절한 폴더 및 파일 구조로 정리.

5. **데이터셋 검증 및 활용**
   - 데이터의 품질과 다양성을 확인.
   - 모델을 학습시키고 결과 평가.

---

## **1. 목적 정의 및 데이터 수집**
### **(1) 목적 정의**
Vision dataset을 만드는 첫 단계는 **목표**를 명확히 하는 것입니다. 예를 들어:

- **객체 탐지(Object Detection)**: 도로에서 차량을 탐지하는 데이터셋
- **이미지 분류(Classification)**: 고양이와 개를 분류하는 데이터셋
- **세그멘테이션(Segmentation)**: 사람과 배경을 분리하는 데이터셋
- **행동 인식(Action Recognition)**: 스포츠 경기에서 특정 동작을 탐지하는 데이터셋

### **(2) 데이터 수집 방법**
데이터를 수집하는 방법은 다음과 같습니다.

#### **① 직접 촬영**
- 스마트폰이나 카메라로 직접 촬영한 이미지/동영상을 활용.
- 다양한 환경(조명, 각도, 거리)에서 촬영해야 일반화된 모델을 만들 수 있음.

#### **② 웹 크롤링(Web Scraping)**
- **Google Images, Bing, Unsplash** 등에서 이미지를 다운로드.
- **Python의 `BeautifulSoup` 및 `Selenium`** 등을 사용하여 자동 수집 가능.
- **주의**: 저작권 문제가 없는 데이터만 사용.

#### **③ 공개 데이터셋 활용**
- 오픈소스 데이터셋을 활용하면 시간 절약 가능.
- 대표적인 공개 데이터셋:
  - **COCO (Common Objects in Context)**: 객체 탐지 및 세그멘테이션
  - **ImageNet**: 이미지 분류
  - **Open Images Dataset**: Google이 제공하는 대규모 데이터셋

#### **④ 합성 데이터(Synthetic Data)**
- **GAN(Generative Adversarial Networks)** 또는 **3D 렌더링**을 사용하여 생성.
- 특히 의료 이미지(CT, MRI)나 드론 영상에서 활용.

---

## **2. 데이터 전처리**
### **(1) 데이터 정리**
수집한 데이터는 정리 과정이 필요합니다.

- **중복 제거**: 중복 이미지를 삭제 (`hash` 기반 검사 가능).
- **파일 포맷 변환**: JPEG, PNG, TIFF 등 원하는 형식으로 변환.
- **이미지 크기 조정**: 신경망의 입력 크기에 맞게 크롭 및 리사이징.
- **이상치 제거**: 너무 어둡거나 흐릿한 이미지 제거.

```python
from PIL import Image
import os

input_folder = "raw_images/"
output_folder = "processed_images/"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    img = Image.open(os.path.join(input_folder, file))
    img = img.resize((224, 224))  # 모델에 맞게 리사이징
    img.save(os.path.join(output_folder, file))
```

### **(2) 데이터 라벨링**
라벨링(Labeling)은 Vision Dataset에서 가장 중요한 과정 중 하나입니다.

#### **① 이미지 분류(Classification)**
- 단순히 폴더 구조를 사용하여 라벨링.

```
dataset/
 ├── train/
 │   ├── cat/  # 고양이 사진 폴더
 │   ├── dog/  # 개 사진 폴더
 ├── val/
 │   ├── cat/
 │   ├── dog/
```

#### **② 객체 탐지(Object Detection)**
- 각 이미지에 객체의 위치(Bounding Box)를 지정해야 함.
- 대표적인 라벨링 도구:
  - [LabelImg](https://github.com/heartexlabs/labelImg): Bounding Box 라벨링
  - [Roboflow](https://roboflow.com/): 여러 형식 지원 및 자동 라벨링 기능 제공

#### **③ 이미지 세그멘테이션(Segmentation)**
- 픽셀 단위로 객체를 분류해야 함.
- 대표적인 라벨링 도구:
  - [CVAT](https://cvat.org/)
  - [Supervisely](https://supervise.ly/)

---

## **3. 데이터 증강(Augmentation)**
데이터셋이 부족하거나 다양성을 높이기 위해 데이터 증강을 수행합니다.

- **회전 (Rotation)**
- **수평/수직 반전 (Flip)**
- **밝기 및 색상 조정 (Brightness, Contrast)**
- **노이즈 추가 (Noise Injection)**

```python
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

img = Image.open("sample.jpg")
augmented_img = transform(img)
```

---

## **4. 데이터셋 구조화 및 저장**
데이터를 **훈련(train), 검증(val), 테스트(test)** 세트로 분할해야 합니다.

- **훈련 데이터(Train)**: 70~80%
- **검증 데이터(Validation)**: 10~20%
- **테스트 데이터(Test)**: 10%

### **(1) 폴더 구조 예시**
```
dataset/
 ├── train/
 │   ├── images/
 │   ├── labels/
 ├── val/
 │   ├── images/
 │   ├── labels/
 ├── test/
 │   ├── images/
 │   ├── labels/
```

### **(2) CSV 또는 JSON 형식 저장**
객체 탐지 및 세그멘테이션 데이터는 JSON 또는 CSV로 저장.

**COCO JSON 예시**
```json
{
  "images": [{"id": 1, "file_name": "image1.jpg"}],
  "annotations": [{"image_id": 1, "bbox": [100, 200, 50, 50], "category_id": 1}],
  "categories": [{"id": 1, "name": "cat"}]
}
```

---

## **5. 데이터셋 검증 및 활용**
### **(1) 데이터셋 확인**
수집한 데이터셋이 제대로 만들어졌는지 검토.

```python
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("dataset/train/images/sample.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

### **(2) 모델 학습 및 평가**
데이터가 준비되었으면, PyTorch 또는 TensorFlow를 사용하여 모델을 학습.

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=False, num_classes=2)
```

---

## **결론**
Vision Dataset을 만드는 과정은 **목표 설정 → 데이터 수집 → 전처리 → 증강 → 검증**의 단계를 거칩니다. 각 단계에서 적절한 도구와 방법을 선택하면 효율적인 데이터셋을 구축할 수 있습니다.

추가적으로 필요한 부분이 있으면 질문해주세요!
