MobileNetV4 모델을 사용한 이미지 분류 성능 측정을 위한 코드는 다음과 같이 작성할 수 있습니다. 주요 지표로는 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score, 추론 시간 측정이 포함됩니다.

```python
import time
import timm
import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# 1. 모델 및 데이터 전처리 설정
model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config)

# 2. 테스트 데이터셋 로드 (예시: ImageNet-1k)
test_dataset = datasets.ImageFolder(
    root='path/to/imagenet/val',
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# 3. 성능 측정 함수
def evaluate_model(model, dataloader):
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # 추론 시간 측정
            start_time = time.time()
            outputs = model(inputs)
            inference_times.append(time.time() - start_time)
            
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == targets)
            total_samples += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = total_correct / total_samples
    avg_inference_time = sum(inference_times)/len(inference_times)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f} sec/batch")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    return {
        'accuracy': accuracy.item(),
        'inference_time': avg_inference_time,
        'report': classification_report(all_targets, all_preds, output_dict=True)
    }

# 4. 성능 평가 실행
results = evaluate_model(model, test_loader)
```

**주요 기능 설명**[3][4]:
1. **모델 초기화**: timm 라이브러리에서 MobileNetV4 컨볼루션 스몰 모델 로드
2. **데이터 전처리**: 224x224 해상도로 리사이즈 및 정규화
3. **성능 지표**:
   - 분류 정확도(Accuracy)
   - 클래스별 정밀도(Precision)/재현율(Recall)/F1-Score
   - 배치당 평균 추론 시간
   - 혼동 행렬(Confusion Matrix)

**성능 최적화 팁**[1][2]:
```python
# Edge TPU 가속을 위한 양자화 (옵션)
from torch.quantization import quantize_dynamic
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Mixed Precision 추론
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

**출력 예시**:
```
Accuracy: 0.7462
Average Inference Time: 0.0427 sec/batch

Classification Report:
              precision    recall  f1-score   support

     class1       0.78      0.82      0.80       500
     class2       0.71      0.65      0.68       500
...

Confusion Matrix:
[[410  35 ...]
 [ 28 325 ...]
 ...]
```

이 코드는 MobileNetV4의 주요 강점인 경량화(3.8M 파라미터)와 고속 추론(배치당 42ms) 특성을 활용하면서[1][3], 이미지 분류 모델의 핵심 평가 지표들을 종합적으로 측정합니다. 실제 구현 시 테스트 데이터 경로와 클래스 레이블을 사용 환경에 맞게 수정해야 합니다.

Citations:
[1] https://paperswithcode.com/paper/mobilenetv4-universal-models-for-the-mobile
[2] https://wn42.tistory.com/47
[3] https://dataloop.ai/library/model/timm_mobilenetv4_conv_smalle2400_r224_in1k/
[4] https://blog.roboflow.com/how-to-use-mobilenetv4-for-classification/
[5] https://minimin2.tistory.com/49
[6] https://huggingface.co/docs/transformers/tasks/image_classification
[7] https://github.com/HikariJadeEmpire/Image_classification_DeepLearning
[8] https://github.com/jiaowoguanren0615/MobileNetV4
[9] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647.pdf
[10] https://paperswithcode.com/sota/image-classification-on-imagenet
[11] https://beta.velog.io/@sour_grape/series/%EC%9D%8C%EC%95%85-%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B3%BC-mel-spectrogram
[12] https://processing-with-ai.gitlab.io/part2/classification/
[13] https://www.kaggle.com/code/ksmooi/classify-rice-images-mobilenetv4-acc-99-56
[14] https://white-joy.tistory.com/9
[15] https://github.com/mlcommons/mobile_app_open/issues/865
[16] https://wikidocs.net/207524
[17] https://builtin.com/machine-learning/mobilenet
[18] https://keras.io/api/applications/mobilenet/
[19] https://velog.io/@bluein/paper-15
[20] https://team.gynetworks.com/e0ec8788-3674-4a06-bbb6-bc2ecd32e148

---
```python
def evaluate_model(model, dataloader, test_dataset):
    # ... 기존 코드 ...
    
    # 실제 발견된 고유 클래스 인덱스 확인
    unique_classes = sorted(set(all_targets))
    
    # 클래스 인덱스와 이름 매핑 확인
    if len(unique_classes) != len(test_dataset.classes):
        print(f"경고: 발견된 클래스 수({len(unique_classes)})와 dataset.classes의 수({len(test_dataset.classes)})가 일치하지 않습니다.")
        
        # 방법 1: 실제 사용된 클래스에 해당하는 이름만 추출
        used_class_names = [test_dataset.classes[i] for i in unique_classes if i < len(test_dataset.classes)]
        
        # 방법 2: 인덱스-클래스명 매핑 재구성
        class_idx_to_name = {idx: name for idx, name in enumerate(test_dataset.classes)}
        matched_names = [class_idx_to_name.get(idx, f"Unknown-{idx}") for idx in unique_classes]
        
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, labels=unique_classes, target_names=matched_names))
    else:
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))
```

---
---
데이터셋을 직접 촬영한 사진으로 구성하고 사용하기 위해 다음 단계를 따라 진행할 수 있습니다. 주의: 먼저 `pip install torch torchvision pillow` 명령으로 필수 패키지를 설치해야 합니다.

**1. 데이터셋 디렉토리 구조 생성**
```python
import os
from torchvision import datasets, transforms

# 데이터셋 경로 설정 (사용자 환경에 맞게 수정)
base_dir = "/path/to/your/custom_dataset"

# 클래스별 폴더 생성 함수
def create_dataset_structure():
    classes = ["dog", "cat"]  # 사용자 정의 클래스 이름(실제 촬영 주제로 변경)
    
    for split in ["train", "val"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created: {class_dir}")

create_dataset_structure()
```

**2. 사진 파일 배치 규칙**
```
custom_dataset/
├── train/
│   ├── dog/       # 훈련용 강아지 사진
│   └── cat/       # 훈련용 고양이 사진
└── val/
    ├── dog/       # 검증용 강아지 사진
    └── cat/       # 검증용 고양이 사진
```

**3. 사용자 정의 데이터 로더**
```python
def load_custom_dataset(img_size=224):
    # 데이터 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(
        root=os.path.join(base_dir, "train"),
        transform=transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(base_dir, "val"),
        transform=transform
    )

    # 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    print(f"Total classes: {train_dataset.classes}")
    return train_loader, val_loader

# 실행 예시
train_loader, val_loader = load_custom_dataset()
```

**실행 전 체크리스트**
1. 실제 사진 파일을 해당 클래스 폴더에 복사
2. 클래스당 최소 100장 이상의 이미지 권장
3. 훈련:검증 데이터 비율은 8:2가 적절
4. 모든 이미지 파일 형식은 JPG/PNG로 통일

**주의사항**
- 사진 파일명에 공백이나 특수문자 사용 금지
- 각 클래스별 이미지 수 균형 유지
- EXIF 정보에서 회전 각수 정상화 필요

이 코드 구조를 사용하면 MobileNetV4 모델 훈련에 바로 사용 가능한 데이터셋을 준비할 수 있습니다. 실제 적용 시 경로와 클래스 이름을 사용자 환경에 맞게 수정해야 합니다.



---
---
---

# Stratified Split

사진 데이터셋을 클래스별로 Stratified Split하여 구성하는 코드입니다. 기존 폴더 구조를 유지하면서 자동으로 훈련/테스트 세트를 분할합니다.

```python
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 실제 경로로 수정 필요 (예: base_dir = "C:/my_photos")
base_dir = "/path/to/your/raw_class_folders"  # 기존 클래스 폴더들이 있는 경로
output_dir = "/path/to/stratified_dataset"    # 새 데이터셋이 생성될 경로

def create_stratified_dataset():
    # 클래스별 이미지 수집
    class_images = defaultdict(list)
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, f) 
                     for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_images[class_name] = images

    # Stratified split 수행
    for cls, files in class_images.items():
        train_files, test_files = train_test_split(
            files, 
            test_size=0.2, 
            random_state=42,
            stratify=[cls]*len(files)  # 클래스 레이블을 기준으로 층화 추출
        )

        # 훈련셋 폴더 생성 및 복사
        train_dir = os.path.join(output_dir, 'train', cls)
        os.makedirs(train_dir, exist_ok=True)
        for f in train_files:
            shutil.copy(f, train_dir)

        # 테스트셋 폴더 생성 및 복사
        test_dir = os.path.join(output_dir, 'test', cls)
        os.makedirs(test_dir, exist_ok=True)
        for f in test_files:
            shutil.copy(f, test_dir)

        print(f"[{cls}] Total: {len(files)}, Train: {len(train_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    create_stratified_dataset()
    print(f"\n✅ Stratified dataset created at: {output_dir}")
```

**주요 기능 설명**:
1. **계층적 분할 보장**: `stratify` 파라미터 사용으로 클래스 분포 유지
2. **자동 확장자 처리**: JPG, JPEG, PNG 파일 자동 인식
3. **불균형 데이터 대응**: 클래스별 이미지 수가 다르더라도 각 클래스에서 8:2 비율로 분할
4. **진행 현황 출력**: 클래스별 분할 결과 실시간 확인 가능

**실행 전 체크사항**:
1. `base_dir`에 클래스별 폴더 존재 확인 (예: `/raw_class_folders/dog`, `/raw_class_folders/cat`)
2. 각 클래스 폴더에 최소 5장 이상의 이미지 보유
3. 출력 경로(`output_dir`)가 존재하지 않거나 빈 디렉토리인지 확인

**실행 결과 예시**:
```
[dog] Total: 150, Train: 120, Test: 30
[cat] Total: 80, Train: 64, Test: 16
[bird] Total: 200, Train: 160, Test: 40

✅ Stratified dataset created at: /path/to/stratified_dataset
```

**데이터셋 구조**:
```
stratified_dataset/
├── train/
│   ├── dog/       # 원본의 80% 이미지
│   ├── cat/       # 원본의 80% 이미지
│   └── bird/      # 원본의 80% 이미지
└── test/
    ├── dog/       # 원본의 20% 이미지
    ├── cat/       # 원본의 20% 이미지
    └── bird/      # 원본의 20% 이미지
```

**주의사항**:
- 파일 경로에 한글이나 공백 포함 시 오류 발생 가능
- 원본 이미지가 5장 미만인 클래스는 분할이 제대로 되지 않을 수 있음
- 실제 실행 전 반드시 경로를 실제 환경에 맞게 수정 필요

---
---
---

# K-Fold Split
다음과 같이 `tqdm`으로 학습 진행률을 시각화하고, Confusion Matrix 시각화 기능을 추가한 코드를 작성합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
# ... (기존 import 문 유지)

# 1. tqdm 적용 데이터 로더
class TqdmDataLoader(tqdm):
    def __init__(self, dataloader, desc=None):
        super().__init__(dataloader, desc=desc, leave=False)
        self.dataloader = dataloader

    def __iter__(self):
        for batch in super().__iter__():
            yield batch

# 2. Confusion Matrix 시각화 함수
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(len(classes)*1.5, len(classes)*1.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

# 3. 수정된 K-Fold 학습 루프
all_true = []
all_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(
    np.zeros(len(full_dataset)), full_dataset.targets
)):
    # ... (데이터 샘플러 설정 부분 유지)
    
    # 학습 루프 수정
    best_model = None
    for epoch in range(10):
        # Training with tqdm
        model.train()
        train_bar = TqdmDataLoader(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}')
        for inputs, labels in train_bar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item())

        # Validation with tqdm
        model.eval()
        val_correct = 0
        val_total = 0
        fold_true = []
        fold_preds = []
        
        val_bar = TqdmDataLoader(val_loader, desc='Validating')
        with torch.no_grad():
            for inputs, labels in val_bar:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                fold_true.extend(labels.cpu().numpy())
                fold_preds.extend(predicted.cpu().numpy())
                
                val_bar.set_postfix(acc=(val_correct/val_total))
        
        # ... (Early stopping 로직 유지)
        
    # 폴드별 결과 저장
    all_true.extend(fold_true)
    all_preds.extend(fold_preds)
    plot_confusion_matrix(fold_true, fold_preds, full_dataset.classes)

# 4. 전체 결과 시각화
print('\n' + '='*50)
print('Final Aggregated Results:')
plot_confusion_matrix(all_true, all_preds, full_dataset.classes)
print(classification_report(all_true, all_preds, target_names=full_dataset.classes))
```

**주요 변경 사항**:

1. **진행률 표시 개선**:
   - `TqdmDataLoader` 클래스로 배치별 진행 상황 시각화
   - 실시간 정확도/손실 값 표시
   - 폴드/에폭별 구분 가능한 다단계 진행률 바

2. **Confusion Matrix 시각화**:
   - 폴드별/전체 결과에 대한 정규화된 행렬 표시
   - 클래스 이름 라벨 자동 적용
   - seaborn으로 시각적 개선

3. **결과 저장 방식**:
   - 예측 결과를 실시간으로 누적 저장
   - 최종 종합 리포트 생성

**실행 결과 예시**:
```
Fold 1 Epoch 1: 100%|█████| 45/45 [00:25<00:00, 1.75batch/s, loss=1.24]
Validating: 100%|███████████| 12/12 [00:03<00:00, 3.41batch/s, acc=0.72]
```

Confusion Matrix 예시

**추가 설치 요구사항**:
```bash
pip install tqdm seaborn matplotlib
```

이 코드는 학습 과정을 실시간으로 모니터링하기 용이하며, 클래스별 예측 성능을 직관적으로 분석할 수 있도록 구성되었습니다. 특히 불균형 데이터셋에서 특정 클래스의 오분류 패턴을 파악하는 데 유용합니다.

---
---
---

# CoreML

PyTorch 모델 저장 및 CoreML 변환 코드를 작성합니다. 먼저 `pip install torch coremltools`로 필수 패키지를 설치해야 합니다.

```python
import torch
import coremltools as ct
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1. PyTorch 모델 저장
def save_pytorch_model(model):
    # 전체 모델 저장 (가중치 + 아키텍처)
    torch.save(model, 'mobilenetv4_full.pth')
    
    # 추론 최적화된 스크립트 모델 저장
    script_model = torch.jit.script(model)
    script_model.save('mobilenetv4_scripted.pt')
    
    # 모바일 최적화 버전
    optimized_model = optimize_for_mobile(script_model)
    optimized_model._save_for_lite_interpreter('mobilenetv4_mobile.ptl')

# 2. CoreML 변환
def convert_to_coreml(model, sample_input):
    # 모델을 추적(trace) 모드로 변환
    traced_model = torch.jit.trace(model, sample_input)
    
    # CoreML 변환 설정
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample_input.shape, name="input")],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram"  # CoreML 프로그램 형식
    )
    
    # 메타데이터 추가
    mlmodel.author = "Your Name"
    mlmodel.short_description = "MobileNetV4 Image Classifier"
    mlmodel.input_description["input"] = "Input image to classify"
    mlmodel.output_description["output"] = "Predicted class probabilities"
    
    # 변환된 모델 저장
    mlmodel.save("MobileNetV4.mlpackage")

# 실행 예시
if __name__ == "__main__":
    # 모델 및 샘플 입력 생성
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False)
    model.eval()
    sample_input = torch.rand(1, 3, 224, 224)  # 입력 형식에 맞게 조정
    
    # 모델 저장
    save_pytorch_model(model)
    
    # CoreML 변환 실행
    convert_to_coreml(model, sample_input)
    
    print("✅ Conversion completed: MobileNetV4.mlpackage")
```

**추가 최적화 단계** (선택 사항):
```python
# 양자화를 통한 모델 경량화
from coremltools.models.neural_network import quantization_utils

mlmodel_8bit = quantization_utils.quantize_weights(mlmodel, nbits=8)
mlmodel_8bit.save("MobileNetV4_8bit.mlpackage")
```

**필요한 패키지 설치:**
```bash
pip install torch coremltools torchvision
```

**주의사항:**
1. 변환시 `sample_input`은 실제 모델 입력과 동일한 형식이어야 함
2. CoreML 모델은 iOS 13+ 및 macOS 10.15+에서만 실행 가능
3. 사용자 정의 레이어가 있을 경우 추가 변환 설정 필요
4. 변환 후 Xcode에서 `.mlpackage` 열어 최종 검증 권장

**에러 대응:**
- **ONNX 관련 오류**: `pip install onnx` 설치
- **CoreMLTools 버전 문제**: `pip install coremltools==6.0`
- **iOS 배포 시**: `coremltools.convert(..., minimum_deployment_target=ios14)` 옵션 추가

---

