# MobileCLIP

아래는 MobileCLIP 모델을 PyTorch에서 ONNX를 거쳐 CoreML로 변환하는 전체 과정과, 최종적으로 iOS Swift 코드에서 해당 CoreML 모델을 활용하여 classification을 수행하는 방법에 대한 요약 및 자세한 설명입니다.

---

## 요약
1. **PyTorch → ONNX 변환:**  
   - **무엇을 하는가:** PyTorch의 동적 모델을 정적 계산 그래프로 변환하여 ONNX 파일로 저장합니다.  
   - **핵심 코드:** `torch.onnx.export()` 함수를 사용하며, 더미 입력(dummy input)을 통해 모델의 입력/출력 텐서의 shape와 dynamic axis 등을 지정합니다.

2. **ONNX → CoreML 변환:**  
   - **무엇을 하는가:** ONNX 모델을 CoreML의 `.mlmodel` 파일로 변환합니다.  
   - **핵심 코드:** Python의 [onnx-coreml](https://github.com/onnx/onnx-coreml) 라이브러리를 사용하여 변환하며, ONNX의 각 연산자(op)를 CoreML의 해당 레이어로 매핑합니다.

3. **iOS Swift에서 CoreML 사용하여 분류 수행:**  
   - **무엇을 하는가:** Xcode 프로젝트에 `.mlmodel` 파일을 추가하면 자동으로 Swift 클래스가 생성되고, 이를 통해 입력 데이터를 전처리한 후 모델 예측(prediction)을 수행합니다.  
   - **핵심 코드:** 생성된 모델 클래스(예: `MobileCLIP`)의 `prediction` 메소드를 호출하여 이미지(또는 다른 입력 데이터)에 대해 classification 결과를 얻습니다.

---

## 자세한 설명

### 1. PyTorch 모델을 ONNX로 변환하기

#### (1) 모델 준비 및 더미 입력 생성
먼저, PyTorch로 구현된 MobileCLIP 모델을 로드하거나 초기화합니다. CoreML 변환을 위해 모델은 고정된 입력 크기를 필요로 하므로, 더미 입력(dummy input)을 생성하여 모델의 추론 그래프를 고정합니다.

예시 코드:
```python
import torch
import torch.nn as nn

# 예시: MobileCLIP 모델을 정의하거나 불러오기 (여기서는 간단한 예시)
class MobileCLIP(nn.Module):
    def __init__(self):
        super(MobileCLIP, self).__init__()
        # 실제 MobileCLIP은 훨씬 복잡하겠지만, 예시로 간단한 CNN을 사용합니다.
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 112 * 112, 1000)  # 예를 들어 1000 클래스 분류
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 모델 인스턴스 생성 및 평가 모드 설정
model = MobileCLIP()
model.eval()

# 더미 입력: MobileCLIP의 입력이 224x224 크기의 3채널 이미지라고 가정
dummy_input = torch.randn(1, 3, 224, 224)
```

#### (2) ONNX 파일로 내보내기
PyTorch의 `torch.onnx.export()` 함수를 사용하여 모델을 ONNX 형식으로 내보냅니다. 이때 입력 및 출력 이름, 동적 배치 크기(dynamic axes) 등을 지정합니다.

예시 코드:
```python
import torch.onnx

onnx_file_path = "mobileclip.onnx"

torch.onnx.export(
    model,                   # 변환할 PyTorch 모델
    dummy_input,             # 더미 입력
    onnx_file_path,          # 저장할 ONNX 파일 경로
    export_params=True,      # 모델 파라미터도 내보냄
    opset_version=11,        # ONNX opset 버전 (필요에 따라 최신 버전 사용)
    do_constant_folding=True,  # 상수 폴딩을 수행하여 최적화
    input_names=['input'],     # 모델 입력의 이름 지정
    output_names=['output'],   # 모델 출력의 이름 지정
    dynamic_axes={
        'input': {0: 'batch_size'},   # 배치 차원 동적 처리
        'output': {0: 'batch_size'}
    }
)
```
**변환 과정 설명:**  
- PyTorch의 동적 연산 그래프를 정적 계산 그래프로 "트레이싱(tracing)" 또는 "스크립팅(scripting)"하여 ONNX 형식의 파일로 저장합니다.  
- 모델 내부의 연산(예: Conv, ReLU, Linear 등)이 ONNX의 표준 노드(node)로 매핑됩니다.  
- 만약 MobileCLIP에 특별한 커스텀 연산이 있다면, 추가적인 핸들러나 커스텀 레이어 등록이 필요할 수 있습니다.

---

### 2. ONNX 모델을 CoreML 모델로 변환하기

CoreML은 iOS에서 머신러닝 모델을 효율적으로 실행할 수 있도록 최적화된 형식입니다. ONNX에서 CoreML로 변환하는 주요 작업은 각 ONNX 노드를 CoreML의 레이어로 매핑하는 것입니다.

#### (1) onnx-coreml 라이브러리 설치
Python 환경에서 [onnx-coreml](https://github.com/onnx/onnx-coreml) 라이브러리를 설치합니다.
```bash
pip install onnx-coreml
```

#### (2) ONNX 모델을 CoreML 모델로 변환하는 코드
아래 코드는 ONNX 모델 파일을 읽어 CoreML 모델로 변환한 후, `.mlmodel` 파일로 저장하는 예시입니다.
```python
import onnx
from onnx_coreml import convert

# ONNX 모델 로드
onnx_model = onnx.load("mobileclip.onnx")

# 변환 옵션 설정: minimum_ios_deployment_target는 iOS 최소 지원 버전을 지정합니다.
coreml_model = convert(
    model=onnx_model,
    minimum_ios_deployment_target='13'  # 예: iOS 13 이상
)

# 변환된 CoreML 모델 저장
coreml_model.save("MobileCLIP.mlmodel")
```
**변환 과정 설명:**  
- **노드 매핑:** onnx-coreml 라이브러리는 ONNX 모델의 각 연산자(op)를 분석한 후, CoreML에서 지원하는 대응 레이어로 매핑합니다. 예를 들어, Conv2D, BatchNorm, ReLU 등은 CoreML에서 직접 지원하는 레이어로 변환됩니다.  
- **동적/고정 크기:** CoreML은 일반적으로 고정된 입력 크기를 요구합니다. 변환 과정에서 입력 크기가 고정되어 있거나, 일부 동적 축이 변환될 수 있으니 변환 시 주의해야 합니다.  
- **지원하지 않는 연산:** 만약 ONNX 모델에 CoreML에서 기본적으로 지원하지 않는 연산자가 있다면, 사용자 정의 커스텀 레이어(custom layer)를 추가해야 합니다.

---

### 3. iOS Swift에서 CoreML 모델을 사용하여 Classification 수행하기

변환된 `MobileCLIP.mlmodel` 파일을 Xcode 프로젝트에 추가하면, Xcode가 자동으로 Swift 클래스(예: `MobileCLIP` 클래스)를 생성합니다. 이 클래스의 인터페이스를 사용하여 모델 예측을 수행할 수 있습니다.

#### (1) Xcode 프로젝트에 모델 추가
- Xcode에서 해당 `.mlmodel` 파일을 프로젝트 내에 추가합니다.
- Xcode가 이를 컴파일하여 `.mlmodelc` 형태로 변환하고, 자동 생성된 Swift 인터페이스(클래스)를 제공합니다.

#### (2) Swift 코드 예시: 이미지 분류 수행
아래는 iOS 애플리케이션 내에서 CoreML 모델을 사용하여 이미지를 분류하는 예시 코드입니다. 이 예제에서는 입력 이미지가 `UIImage` 형태로 주어지고, 이를 `CVPixelBuffer`로 변환한 후 모델에 전달합니다.

```swift
import UIKit
import CoreML

// CoreML 모델 클래스가 'MobileCLIP'라고 가정
class ImageClassifier {
    // MobileCLIP 모델의 인스턴스 생성
    let model = MobileCLIP()

    // 이미지 분류 메서드
    func classify(image: UIImage) -> String? {
        // 이미지 전처리: 모델이 224x224 크기를 요구한다고 가정합니다.
        guard let resizedImage = image.resized(to: CGSize(width: 224, height: 224)),
              let pixelBuffer = resizedImage.toCVPixelBuffer() else {
            print("이미지 전처리에 실패하였습니다.")
            return nil
        }
        
        do {
            // 모델 예측 수행
            let predictionOutput = try model.prediction(input: pixelBuffer)
            
            // 예측 결과 처리: 예시에서는 'output'이라는 프로퍼티에 결과가 담긴다고 가정합니다.
            // 실제 모델 인터페이스에 따라 달라질 수 있습니다.
            let classificationResult = predictionOutput.output
            
            // 결과를 적절한 방식(예: 문자열 변환 등)으로 반환
            return "\(classificationResult)"
        } catch {
            print("모델 예측 중 에러 발생: \(error)")
            return nil
        }
    }
}

// UIImage를 CVPixelBuffer로 변환하기 위한 확장 예시
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(self.size.width),
                                         Int(self.size.height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(self.size.width),
                                      height: Int(self.size.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        context.translateBy(x: 0, y: self.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
```

**Swift 코드 설명:**  
- **이미지 전처리:**  
  - 입력 이미지(`UIImage`)를 모델이 요구하는 크기(예, 224×224)로 리사이즈합니다.  
  - 리사이즈된 이미지를 `CVPixelBuffer`로 변환합니다. CoreML은 보통 `CVPixelBuffer` 또는 `MLMultiArray` 형태의 입력을 요구합니다.
  
- **모델 예측:**  
  - 자동 생성된 `MobileCLIP` 클래스의 인스턴스를 생성하고, `prediction(input:)` 메서드를 통해 분류 작업을 수행합니다.
  - 예측 결과는 모델에 따라 다양한 형식(예: 클래스 확률, 레이블 등)으로 반환되며, 이를 적절히 처리합니다.
  
- **오류 처리:**  
  - 예측 과정에서 오류가 발생할 경우, 이를 캐치하여 로그를 남기고 nil 또는 적절한 오류 값을 반환합니다.

---

## 참고 자료
- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)  
- [ONNX-CoreML GitHub Repository](https://github.com/onnx/onnx-coreml)  
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)  

이와 같이 MobileCLIP 모델을 PyTorch에서 ONNX로 변환한 후, CoreML로 다시 변환하여 iOS 애플리케이션 내에서 활용할 수 있습니다. 각 단계마다 모델의 구조와 연산자들이 적절하게 매핑되어야 하며, 특히 ONNX에서 CoreML로의 변환 시 지원되지 않는 연산자에 대해서는 추가적인 처리가 필요할 수 있습니다. 추가적인 문의 사항이 있으시면 언제든지 질문해 주세요.

# Vit

아래는 Vision Transformer(ViT) 모델을 PyTorch에서 ONNX를 거쳐 CoreML로 변환하는 전체 과정과, 최종적으로 iOS 애플리케이션에서 CoreML 모델을 사용하여 분류(classification)를 수행하는 방법에 대해 자세하게 설명드리겠습니다.

---

## 요약

1. **PyTorch에서 ViT 모델 준비 및 ONNX 변환**  
   - **모델 준비:** ViT의 주요 구성요소(패치 임베딩, [CLS] 토큰, 포지셔널 임베딩, Transformer 인코더, 분류 헤드)를 포함하는 간단한 ViT 모델을 구현합니다.  
   - **더미 입력:** 모델의 입력 크기(예, 224×224×3 이미지)를 기반으로 더미 입력을 생성하여 모델의 추론 그래프를 고정합니다.  
   - **ONNX 변환:** `torch.onnx.export()` 함수를 사용하여 모델을 ONNX 파일로 내보내며, 입력/출력 이름과 동적 축(dynamic axes) 등을 지정합니다.

2. **ONNX 모델을 CoreML 모델로 변환**  
   - **변환 라이브러리:** Python의 [onnx-coreml](https://github.com/onnx/onnx-coreml) 라이브러리를 사용하여 ONNX 모델을 CoreML 모델(.mlmodel)로 변환합니다.  
   - **매핑:** ONNX의 연산자(operator)들이 CoreML에서 지원하는 레이어로 매핑되며, 일부 연산자의 경우 사용자 정의 커스텀 레이어를 추가해야 할 수 있습니다.

3. **iOS Swift에서 CoreML 모델로 분류 수행**  
   - **Xcode 통합:** 변환된 `.mlmodel` 파일을 Xcode 프로젝트에 추가하면 자동으로 Swift 인터페이스(클래스)가 생성됩니다.  
   - **모델 사용:** Swift 코드에서 입력 이미지(UIImage)를 전처리하여 `CVPixelBuffer`로 변환한 후, 생성된 모델 클래스의 `prediction` 메서드를 호출하여 분류 결과를 얻습니다.

---

## 1. PyTorch에서 ViT 모델 준비 및 ONNX 변환

### (1) ViT 모델 구현 예시

ViT 모델은 이미지를 일정 크기의 패치로 나눈 뒤, 각 패치를 임베딩하고 Transformer 인코더를 통과시켜 최종적으로 분류하는 구조입니다. 아래는 매우 간단한 ViT 모델 구현 예시입니다.

```python
import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=6, heads=12):
        """
        image_size: 입력 이미지의 크기 (정사각형)
        patch_size: 이미지를 나눌 패치의 크기
        num_classes: 분류 클래스 수
        dim: 패치 임베딩의 차원
        depth: Transformer 인코더 블록 수
        heads: Multi-head self-attention의 헤드 수
        """
        super(SimpleViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 예: 224/16 = 14, 14x14=196 패치
        self.dim = dim
        
        # 패치 임베딩: Conv2d를 이용하여 이미지 패치를 선형 임베딩
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # [CLS] 토큰 (배치마다 학습 가능한 토큰)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 포지셔널 임베딩: 각 패치에 위치 정보를 더해줌 (패치 수 + 1개의 토큰)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Transformer 인코더: nn.TransformerEncoderLayer를 여러 개 쌓음
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 분류 헤드: [CLS] 토큰을 이용해 분류를 수행
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 3, image_size, image_size)
        # 패치 임베딩 수행 → 결과 shape: (batch, dim, H', W') with H' = W' = image_size/patch_size
        x = self.patch_embedding(x)
        # flatten하여 각 패치를 하나의 벡터로 만듦
        x = x.flatten(2)  # shape: (batch, dim, num_patches)
        x = x.transpose(1, 2)  # shape: (batch, num_patches, dim)
        
        batch_size = x.shape[0]
        # [CLS] 토큰을 배치마다 prepend (앞에 추가)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (batch, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (batch, num_patches+1, dim)
        
        # 포지셔널 임베딩을 더함
        x = x + self.pos_embedding
        
        # Transformer 인코더 적용을 위해 transpose: nn.TransformerEncoder는 (seq_len, batch, dim) 입력을 요구함
        x = x.transpose(0, 1)  # shape: (num_patches+1, batch, dim)
        x = self.transformer(x)  # 동일한 shape 유지
        
        # [CLS] 토큰의 최종 출력을 가져와 분류 헤드에 전달
        x = x[0]  # shape: (batch, dim)
        x = self.mlp_head(x)  # shape: (batch, num_classes)
        return x

# 모델 인스턴스 생성 및 평가 모드로 전환
model = SimpleViT()
model.eval()

# 더미 입력: ViT 모델은 일반적으로 224×224 크기의 3채널 이미지를 입력으로 사용
dummy_input = torch.randn(1, 3, 224, 224)
```

### (2) PyTorch 모델을 ONNX로 내보내기

PyTorch의 `torch.onnx.export()` 함수를 사용하여 위에서 구현한 ViT 모델을 ONNX 형식으로 변환합니다. 이때 입력 및 출력 이름, 동적 축(dynamic axes) 등을 지정할 수 있습니다.

```python
import torch.onnx

onnx_file_path = "simple_vit.onnx"

torch.onnx.export(
    model,                   # 변환할 PyTorch ViT 모델
    dummy_input,             # 더미 입력
    onnx_file_path,          # 저장할 ONNX 파일 경로
    export_params=True,      # 모델 파라미터도 함께 내보냄
    opset_version=11,        # ONNX opset 버전 (필요에 따라 최신 버전 사용 가능)
    do_constant_folding=True,  # 상수 폴딩 최적화 수행
    input_names=['input'],     # 입력 이름 지정
    output_names=['output'],   # 출력 이름 지정
    dynamic_axes={
        'input': {0: 'batch_size'},    # 배치 차원을 동적으로 처리
        'output': {0: 'batch_size'}
    }
)
```

**변환 과정 설명:**  
- **트레이싱/스크립팅:** PyTorch 모델의 연산 그래프를 정적으로 추출하여 ONNX 형식으로 저장합니다.  
- **연산자 매핑:** ViT 모델의 연산(예: Conv2d, flatten, Transformer 인코더 등)이 ONNX의 표준 노드로 매핑됩니다.  
- **주의사항:** Transformer 관련 연산이나 동적 시퀀스 길이 등은 변환 시 특별한 주의가 필요할 수 있으며, 복잡한 연산자는 CoreML에서 지원 여부를 확인해야 합니다.

---

## 2. ONNX 모델을 CoreML 모델로 변환

CoreML은 iOS에서 머신러닝 모델을 최적화하여 실행할 수 있는 형식입니다. ONNX 모델을 CoreML 모델로 변환하는 과정에서는 onnx-coreml 라이브러리가 ONNX의 각 연산자를 CoreML 레이어에 맞게 매핑합니다.

### (1) onnx-coreml 라이브러리 설치

터미널에서 아래 명령어로 라이브러리를 설치합니다.

```bash
pip install onnx-coreml
```

### (2) ONNX 모델을 CoreML 모델로 변환하는 코드

아래 코드는 저장된 `simple_vit.onnx` 파일을 읽어 CoreML 모델(`.mlmodel`)로 변환하는 예시입니다.

```python
import onnx
from onnx_coreml import convert

# ONNX 모델 로드
onnx_model = onnx.load("simple_vit.onnx")

# 변환: minimum_ios_deployment_target는 iOS의 최소 지원 버전을 지정합니다.
coreml_model = convert(
    model=onnx_model,
    minimum_ios_deployment_target='13'  # 예: iOS 13 이상 지원
)

# 변환된 CoreML 모델 저장
coreml_model.save("SimpleViT.mlmodel")
```

**변환 과정 설명:**  
- **노드 매핑:** onnx-coreml는 ONNX 모델의 각 노드를 분석하여 CoreML에서 대응하는 레이어로 변환합니다.  
- **입력 크기:** CoreML은 보통 고정된 입력 크기를 요구하므로, 변환 시 입력 텐서의 shape이 고정되어 있어야 합니다.  
- **제한 사항:** 만약 ONNX에서 지원하지만 CoreML에서 기본적으로 지원하지 않는 연산자가 있다면, 추가적인 커스텀 레이어(custom layer) 구현이 필요할 수 있습니다.

---

## 3. iOS Swift에서 CoreML 모델로 분류 수행

변환된 `SimpleViT.mlmodel` 파일을 Xcode 프로젝트에 추가하면, Xcode가 자동으로 Swift 클래스(예, `SimpleViT`)를 생성해 줍니다. 이를 이용해 iOS 애플리케이션 내에서 이미지 분류를 수행할 수 있습니다.

아래에서는 입력 이미지(`UIImage`)를 모델이 요구하는 크기(224×224)로 전처리한 후, `CVPixelBuffer`로 변환하여 CoreML 모델의 예측 메서드를 호출하는 방법을 설명합니다.

---

## Swift 코드 (맨 아랫부분)

```swift
import UIKit
import CoreML

// CoreML 모델 클래스는 'SimpleViT'라는 이름으로 자동 생성되었다고 가정합니다.
class ImageClassifier {
    // SimpleViT 모델의 인스턴스 생성
    let model = SimpleViT()
    
    // 이미지 분류 함수: UIImage를 입력받아 분류 결과(예: 문자열)를 반환
    func classify(image: UIImage) -> String? {
        // 1. 이미지 전처리: 모델이 224x224 크기의 이미지를 요구한다고 가정합니다.
        guard let resizedImage = image.resized(to: CGSize(width: 224, height: 224)),
              let pixelBuffer = resizedImage.toCVPixelBuffer() else {
            print("이미지 전처리에 실패하였습니다.")
            return nil
        }
        
        do {
            // 2. 모델 예측 수행
            let predictionOutput = try model.prediction(input: pixelBuffer)
            
            // 3. 예측 결과 처리: 예시에서는 'output' 프로퍼티에 예측된 클래스 확률 혹은 레이블이 담긴다고 가정합니다.
            let classificationResult = predictionOutput.output
            
            // 예측 결과를 문자열 등으로 변환하여 반환 (구체적인 처리 방식은 모델 인터페이스에 따라 다름)
            return "\(classificationResult)"
        } catch {
            print("모델 예측 중 오류 발생: \(error)")
            return nil
        }
    }
}

// UIImage를 CVPixelBuffer로 변환하기 위한 확장
extension UIImage {
    /// 지정한 크기로 이미지를 리사이즈합니다.
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }
    
    /// UIImage를 CoreML이 요구하는 CVPixelBuffer로 변환합니다.
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(self.size.width),
                                         Int(self.size.height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(self.size.width),
                                      height: Int(self.size.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        // 좌표계를 맞추기 위해 변환
        context.translateBy(x: 0, y: self.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
```

---

## 참고 자료

- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)  
- [ONNX-CoreML GitHub Repository](https://github.com/onnx/onnx-coreml)  
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)  

위와 같이 ViT 모델을 예로 들어 PyTorch에서 ONNX로, 그리고 CoreML로 변환하는 전체 프로세스와 iOS Swift에서의 활용 방법을 단계별로 설명드렸습니다. 각 단계에서 모델의 구조와 연산자들이 올바르게 매핑되어야 하며, 특히 Transformer 관련 연산이나 동적 시퀀스 처리 부분은 변환 시 주의하여 확인하시기 바랍니다.

추가적인 문의 사항이나 궁금한 점이 있으시면 언제든지 질문해 주세요.


# Transformer (Custom)


아래는 복잡한 Transformer 모델의 예시로, PyTorch에서 사용자 정의 커스텀 레이어(예, 입력값을 제곱하는 CustomSquare)를 포함한 Transformer 모델을 ONNX로 내보내고, 이를 CoreML로 변환하는 전체 과정을 설명드립니다.  
또한 iOS에서 이 모델을 사용하기 위해 Swift 코드(및 CoreML의 커스텀 레이어를 구현하는 예제)를 마지막에 제공해 드리겠습니다.

---

## 요약

1. **PyTorch 모델 구현**  
   - **Transformer 모델 구성:** 토큰 임베딩, 위치 임베딩, 여러 Transformer 블록(자기‐어텐션 및 피드포워드 네트워크 포함)으로 구성됩니다.  
   - **사용자 정의 커스텀 레이어:** 피드포워드 네트워크 중 일반 활성화 대신, 입력값을 제곱하는 CustomSquare 레이어를 사용합니다.

2. **ONNX 변환 준비 및 내보내기**  
   - **커스텀 연산자 등록:** PyTorch의 `torch.onnx.register_custom_op_symbolic`을 사용하여 CustomSquare에 대한 심볼릭 함수를 등록합니다.  
   - **torch.onnx.export() 사용:** 모델과 더미 입력을 통해 ONNX 파일로 내보냅니다.

3. **ONNX → CoreML 변환**  
   - **onnx-coreml 라이브러리 사용:** ONNX 모델을 CoreML로 변환할 때, `custom_conversion_functions`를 통해 ONNX의 CustomSquare 연산자를 CoreML의 커스텀 레이어로 매핑합니다.
   - **변환 결과:** 변환된 CoreML 모델은 커스텀 레이어에 대해 iOS에서 MLCustomLayer 프로토콜로 구현된 클래스를 요구합니다.

4. **iOS Swift에서 모델 및 커스텀 레이어 사용**  
   - **CoreML 모델 사용:** Xcode에 추가된 `.mlmodel` 파일을 통해 자동 생성된 모델 클래스를 사용합니다.
   - **커스텀 레이어 구현:** Swift에서 MLCustomLayer 프로토콜을 채택하여 CustomSquareLayer를 구현하고, 모델 예측 시 이를 연결합니다.

---

## 1. PyTorch에서 복잡한 Transformer 모델 구현 (사용자 정의 커스텀 레이어 포함)

### (1) 사용자 정의 커스텀 레이어: CustomSquare  
아래 예제에서는 입력 텐서를 제곱하는 간단한 커스텀 레이어를 정의합니다.
  
```python
import torch
import torch.nn as nn

class CustomSquare(nn.Module):
    def forward(self, x):
        return x * x
```

### (2) CustomSquare에 대한 ONNX 심볼릭 함수 등록  
ONNX 내보내기 시, 사용자 정의 연산자를 처리하기 위해 심볼릭 함수를 등록합니다.  
(주의: 등록은 `torch.onnx.export()` 호출 전에 수행되어야 합니다.)
  
```python
import torch.onnx
from torch.onnx import register_custom_op_symbolic

# opset 버전 11 기준으로 등록 (사용 중인 opset에 맞게 조정)
def custom_square_symbolic(g, input):
    # ONNX 그래프에 "CustomSquare"라는 이름의 노드로 추가합니다.
    return g.op("CustomSquare", input)

register_custom_op_symbolic("::CustomSquare", custom_square_symbolic, 11)
```

### (3) Transformer 블록 및 전체 모델 구현  
Transformer 블록에서는 MultiheadAttention와 피드포워드 네트워크를 포함하는데, 피드포워드 네트워크에서는 일반 활성화 함수 대신 CustomSquare를 사용합니다.

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        # 여기서 사용자 정의 활성화 함수(CustomSquare)를 사용
        self.custom_act = CustomSquare()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        
    def forward(self, x):
        # x: (seq_len, batch, dim)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        ff = self.fc1(x)
        ff = self.custom_act(ff)  # 사용자 정의 커스텀 레이어 적용
        ff = self.fc2(ff)
        
        x = x + ff
        x = self.norm2(x)
        return x
```

전체 모델은 여러 TransformerBlock을 쌓아 구성합니다. 예제에서는 간단한 언어 모델을 가정하여, 토큰 임베딩과 위치 임베딩을 사용합니다.

```python
class ComplexTransformer(nn.Module):
    def __init__(self, vocab_size=10000, seq_length=128, num_layers=4, dim=512, num_heads=8):
        super(ComplexTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # 위치 임베딩: (seq_length, dim)
        self.pos_embedding = nn.Parameter(torch.randn(seq_length, dim))
        # 여러 Transformer 블록을 쌓음
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        # 최종 출력: 각 시퀀스 토큰마다 vocab_size 차원의 로짓 출력
        self.fc_out = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_length) → 토큰 인덱스
        x = self.token_embedding(x)  # (batch, seq_length, dim)
        x = x + self.pos_embedding   # (batch, seq_length, dim)
        # MultiheadAttention는 (seq_len, batch, dim)을 요구하므로 transpose 수행
        x = x.transpose(0, 1)  # (seq_length, batch, dim)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_length, dim)
        x = self.norm(x)
        logits = self.fc_out(x)  # (batch, seq_length, vocab_size)
        return logits

# 모델 인스턴스 생성 및 평가 모드 전환
model = ComplexTransformer()
model.eval()

# 더미 입력: 예를 들어, 배치 크기 1, 시퀀스 길이 128, 정수 토큰 (0 ~ vocab_size-1)
dummy_input = torch.randint(0, 10000, (1, 128))
```

### (4) PyTorch → ONNX 변환

```python
onnx_file_path = "complex_transformer.onnx"

torch.onnx.export(
    model,                    # 변환할 모델
    dummy_input,              # 더미 입력
    onnx_file_path,           # 저장할 ONNX 파일 경로
    export_params=True,       # 학습된 파라미터 포함
    opset_version=11,         # 사용하는 opset 버전 (필요 시 최신 버전으로 조정)
    do_constant_folding=True, # 상수 폴딩 최적화 수행
    input_names=['input'],    # 입력 이름 지정
    output_names=['output'],  # 출력 이름 지정
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'seq_length'},
        'output': {0: 'batch_size', 1: 'seq_length'}
    }
)
```

> **설명:**  
> - 위 과정에서 PyTorch의 동적 연산 그래프를 정적 ONNX 그래프로 변환하며, 커스텀 레이어인 CustomSquare는 미리 등록한 심볼릭 함수에 따라 “CustomSquare” 연산자로 내보내집니다.

---

## 2. ONNX 모델을 CoreML로 변환 (사용자 정의 커스텀 레이어 매핑 포함)

CoreML로 변환할 때, onnx-coreml 라이브러리의 `custom_conversion_functions` 매개변수를 사용하여 ONNX의 “CustomSquare” 연산자를 CoreML 커스텀 레이어로 매핑합니다.

### (1) onnx-coreml 설치

```bash
pip install onnx-coreml
```

### (2) 커스텀 변환 함수 정의 및 ONNX → CoreML 변환 코드

아래 예제에서는 “CustomSquare” 연산자를 CoreML에서 “CustomSquareLayer”라는 이름의 커스텀 레이어로 변환하도록 정의합니다.

```python
import onnx
from onnx_coreml import convert

# 커스텀 변환 함수: ONNX의 "CustomSquare" op를 CoreML 커스텀 레이어로 매핑
def convert_custom_square(layer):
    # layer.parameters에 필요한 추가 매개변수가 있다면 처리 (여기서는 간단히 빈 dict 사용)
    return {
        'className': 'CustomSquareLayer',  # Swift에서 구현할 커스텀 레이어 이름
        'parameters': {}
    }

# 커스텀 변환 함수 딕셔너리: key는 ONNX op 이름 ("CustomSquare")
custom_conversion_functions = {"CustomSquare": convert_custom_square}

# ONNX 모델 로드
onnx_model = onnx.load("complex_transformer.onnx")

# CoreML 모델로 변환 (iOS 최소 배포 타겟 예: iOS 13)
coreml_model = convert(
    model=onnx_model,
    minimum_ios_deployment_target='13',
    custom_conversion_functions=custom_conversion_functions
)

# 변환된 CoreML 모델 저장
coreml_model.save("ComplexTransformer.mlmodel")
```

> **설명:**  
> - 변환 과정에서 onnx-coreml 라이브러리는 ONNX 그래프 내 "CustomSquare" 노드를 발견하면, 제공된 custom_conversion_functions에 따라 이를 CoreML 커스텀 레이어(여기서는 “CustomSquareLayer”)로 매핑합니다.  
> - iOS 앱에서는 이 이름에 맞추어 Swift에서 MLCustomLayer 프로토콜을 구현한 CustomSquareLayer 클래스를 제공해야 합니다.

---

## 3. iOS Swift에서 CoreML 모델 및 커스텀 레이어 사용하기

아래의 Swift 코드는  
1. CoreML 모델(ComplexTransformer.mlmodel)을 사용하여 예측을 수행하는 예시와  
2. 커스텀 레이어인 CustomSquareLayer를 MLCustomLayer 프로토콜로 구현하는 기본 예시를 보여드립니다.

---

### Swift 코드 (맨 아랫부분)

```swift
import UIKit
import CoreML
import MetalPerformanceShaders

// MARK: - 1. CoreML 모델을 사용한 예측 예시

// ComplexTransformer.mlmodel 파일을 Xcode 프로젝트에 추가하면, 자동으로 생성된 모델 클래스(예: ComplexTransformer)가 존재합니다.
class TransformerInference {
    // 생성된 모델 인스턴스
    let model = ComplexTransformer()
    
    /// 입력 토큰 시퀀스(MLMultiArray)를 받아 모델 예측을 수행하는 함수
    func predict(inputSequence: MLMultiArray) -> MLMultiArray? {
        do {
            // 모델 예측 (입력과 출력의 이름은 .mlmodel에 정의된 인터페이스에 따릅니다)
            let prediction = try model.prediction(input: inputSequence)
            return prediction.output
        } catch {
            print("모델 예측 오류: \(error)")
            return nil
        }
    }
}

// MARK: - 2. CoreML 커스텀 레이어 구현: CustomSquareLayer
//
// CoreML 모델 변환 시, ONNX의 "CustomSquare" op는 "CustomSquareLayer"로 매핑되었습니다.
// 이를 위해 iOS 측에서 MLCustomLayer 프로토콜을 채택하는 클래스를 구현해야 합니다.

class CustomSquareLayer: NSObject, MLCustomLayer {
    
    // 초기화: 변환 시 전달되는 파라미터를 처리할 수 있음 (여기서는 사용하지 않음)
    required init(parameters: [String : Any]) throws {
        // 예를 들어, 파라미터가 필요한 경우 처리 가능
        super.init()
    }
    
    // 가중치 데이터 설정 (CustomSquare의 경우 별도의 가중치가 없으므로 빈 배열)
    func setWeightData(_ weights: [Data]) throws {
        // 가중치가 없다면 아무 것도 하지 않음
    }
    
    // 입력 형태에 따른 출력 형태 결정: 입력과 동일한 형태 반환
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return inputShapes
    }
    
    // CPU에서 추론 시 실제 연산 구현: 입력의 각 원소를 제곱하여 출력에 저장
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        // 예를 들어, 첫 번째 입력과 출력만 처리
        guard let input = inputs.first, let output = outputs.first else {
            throw NSError(domain: "CustomSquareLayer", code: -1, userInfo: nil)
        }
        
        let count = input.count
        for i in 0..<count {
            let value = input[i].doubleValue
            output[i] = NSNumber(value: value * value)
        }
    }
    
    // (선택 사항) GPU를 이용한 연산 구현: Metal 커맨드 버퍼와 텍스처를 사용하는 경우 구현
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
        // GPU 가속을 구현하고 싶다면 이곳에 코드를 추가합니다.
        // 여기서는 구현하지 않고 예외 처리합니다.
        throw NSError(domain: "CustomSquareLayer", code: -1, userInfo: [NSLocalizedDescriptionKey: "GPU 인코딩 미구현"])
    }
}
```

---

## 참고 자료

- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)  
- [ONNX-CoreML GitHub Repository](https://github.com/onnx/onnx-coreml)  
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)  
- [MLCustomLayer Protocol Documentation](https://developer.apple.com/documentation/coreml/mlcustomlayer)

---

위와 같이 복잡한 Transformer 모델에 사용자 정의 커스텀 레이어(여기서는 CustomSquare)를 포함한 예시를 통해,  
1. PyTorch에서 모델을 구현하고 ONNX로 변환하는 과정,  
2. ONNX 모델을 CoreML로 변환 시 커스텀 레이어 매핑 처리,  
3. iOS Swift에서 CoreML 모델 및 해당 커스텀 레이어를 사용하는 방법을 자세히 설명드렸습니다.  
추가 문의 사항이나 더 궁금한 점이 있으시면 언제든지 질문해 주세요.
