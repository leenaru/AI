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

