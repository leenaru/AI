### **YOLOv3를 사용하여 직접 촬영한 사진으로 객체 탐지 모델을 파인튜닝하는 방법**

---

## **요약**
1. **환경 설정**: YOLOv3 설치 및 필수 라이브러리 준비  
2. **데이터셋 구축**: 직접 촬영한 사진을 YOLO 형식으로 변환  
3. **구성 파일 수정**: `cfg`, `data`, `names` 파일 설정  
4. **모델 학습**: YOLOv3를 학습시키기  
5. **모델 평가 및 객체 탐지 실행**  
6. **모델 배포**: ONNX 변환, TensorRT 최적화  

---

# **1. 환경 설정**
## **1.1 YOLOv3 설치**
### **방법 1: Ultralytics YOLOv3 (PyTorch)**
```bash
git clone https://github.com/ultralytics/yolov3.git
cd yolov3
pip install -r requirements.txt
```

### **방법 2: Darknet YOLOv3**
```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```

## **1.2 필수 패키지 설치**
```bash
pip install torch torchvision numpy opencv-python tqdm matplotlib
```

---

# **2. 데이터셋 구축**
### **2.1 직접 촬영한 이미지 준비**
카메라로 찍은 사진을 사용하려면 **YOLO 형식의 라벨링**을 해야 합니다.

1. `dataset/images/` 폴더를 만들고, 직접 찍은 사진을 저장합니다.
2. `dataset/labels/` 폴더를 만들고, 각 이미지에 대한 라벨 정보를 저장합니다.

> 💡 **이미지 해상도**: YOLO는 일반적으로 `640x640`, `416x416` 등의 크기를 선호합니다.  
> OpenCV를 사용하여 이미지 크기를 조정할 수 있습니다.

```python
import cv2
import glob
import os

input_folder = "dataset/raw_images"
output_folder = "dataset/images"
os.makedirs(output_folder, exist_ok=True)

for img_path in glob.glob(input_folder + "/*.jpg"):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (640, 640))
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_folder, filename), resized_img)
```

---

### **2.2 라벨링 작업**
#### **방법 1: LabelImg 사용 (GUI 기반)**
LabelImg는 직관적인 GUI를 제공하여 YOLO 형식의 라벨을 생성할 수 있습니다.

1. 설치  
   ```bash
   pip install labelImg
   ```
2. 실행  
   ```bash
   labelImg
   ```
3. `PascalVOC` 형식으로 라벨을 지정한 후, `YOLO` 형식으로 변환 가능

---

#### **방법 2: Roboflow 사용 (온라인 도구)**
Roboflow를 사용하면 웹에서 손쉽게 라벨링을 할 수 있습니다.
- [Roboflow](https://roboflow.com) 가입 후 데이터 업로드
- YOLO 형식으로 변환 후 다운로드

---

### **2.3 YOLO 형식 라벨 예제**
각 이미지에 대한 `.txt` 파일을 `dataset/labels/` 폴더에 저장해야 합니다.

파일 구조:
```
dataset/
 ├── images/
 │   ├── image1.jpg
 │   ├── image2.jpg
 ├── labels/
 │   ├── image1.txt
 │   ├── image2.txt
```
각 `.txt` 파일 내용 예제 (`x_center y_center width height` 값은 **0~1 범위**여야 함):

```
0 0.45 0.55 0.2 0.3
1 0.60 0.50 0.15 0.25
```
- `0` → 클래스 ID
- `0.45` → 객체의 중심 x 좌표 (이미지 너비 기준)
- `0.55` → 객체의 중심 y 좌표 (이미지 높이 기준)
- `0.2` → 객체의 너비
- `0.3` → 객체의 높이

---

# **3. 구성 파일 수정**
### **3.1 클래스 정의 (`dataset/obj.names`)**
YOLO는 객체 클래스를 `.names` 파일에서 관리합니다.

예시 (`dataset/obj.names`):
```
car
person
dog
```

---

### **3.2 데이터 파일 수정 (`dataset/obj.data`)**
```plaintext
classes = 3
train = dataset/train.txt
valid = dataset/val.txt
names = dataset/obj.names
backup = backup/
```

---

### **3.3 `cfg` 파일 수정**
- `yolov3.cfg` 파일을 복사하여 수정
```bash
cp yolov3/cfg/yolov3.cfg yolov3/cfg/custom_yolov3.cfg
```

- `custom_yolov3.cfg`에서 `[yolo]` 레이어 부분 수정
```ini
[convolutional]
filters = (num_classes + 5) * 3  # (3 + 5) * 3 = 24
...
[yolo]
classes = 3
```

---

# **4. 모델 학습**
## **4.1 Darknet 기반 학습**
```bash
./darknet detector train dataset/obj.data yolov3/cfg/custom_yolov3.cfg yolov3.weights -dont_show -map
```

## **4.2 PyTorch 기반 학습**
```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset/obj.data --cfg yolov3/cfg/custom_yolov3.cfg --weights yolov3.weights
```

---

# **5. 모델 평가 및 객체 탐지**
### **5.1 모델 평가**
```bash
python test.py --data dataset/obj.data --weights runs/train/exp/weights/best.pt
```

### **5.2 객체 탐지 실행**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source dataset/images
```

---

# **6. 모델 배포**
### **6.1 ONNX 변환**
```python
import torch

model = torch.load("runs/train/exp/weights/best.pt", map_location="cpu")
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, "yolov3.onnx", opset_version=11)
```

### **6.2 TensorRT 최적화**
```bash
trtexec --onnx=yolov3.onnx --saveEngine=yolov3.trt
```

---

# **마무리**
1. **YOLOv3 설치** (Darknet 또는 PyTorch 버전)
2. **데이터셋 구축**
   - 직접 촬영한 사진 사용
   - LabelImg, Roboflow 등을 활용해 라벨링
3. **구성 파일 수정**
   - `cfg`, `obj.names`, `obj.data` 설정
4. **모델 학습**
   - PyTorch 또는 Darknet에서 학습 실행
5. **모델 평가 및 테스트**
6. **모델 배포**
   - ONNX 변환 및 TensorRT 최적화

이 방법을 따라가면 직접 촬영한 이미지로 YOLOv3를 훈련할 수 있습니다!  
추가 질문이 있으면 언제든지 알려주세요.
